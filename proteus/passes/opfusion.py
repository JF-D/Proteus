from collections import namedtuple, defaultdict
from enum import Enum
from toposort import toposort_flatten
from proteus import OpType, enum_to_int
from proteus.ir import ProteusModel
from proteus.ir.fused_op import FusedOp


class Pattern(Enum):
    Membound = 0
    Compbound = 1
    Opaque = 2

    @staticmethod
    def get(optype):
        if optype in [
                OpType.Linear, OpType.LinearBW, OpType.Conv2d, OpType.Conv2dBW,
                OpType.Matmul, OpType.MatmulBW
        ]:
            return Pattern.Compbound
        elif optype in [OpType.Reshape]:
            return Pattern.Opaque
        return Pattern.Membound

    def __lt__(self, other):
        return enum_to_int(Pattern, self) < enum_to_int(Pattern, other)

    def __le__(self, other):
        return enum_to_int(Pattern, self) <= enum_to_int(Pattern, other)


class IndexedGraph:
    FNode = namedtuple('FNode', ('index', 'pattern', 'outputs', 'orgid'))
    FEdge = namedtuple('FEdge', ('node', 'pattern'))

    def __init__(self, graph: ProteusModel, forward=False):

        self.node_map = {}
        self.topo_order = []

        redges, _ = graph.make_op_graph(forward=forward, reverse=True)
        topo_order = toposort_flatten(redges)

        for idx, nid in enumerate(topo_order):
            self.node_map[nid] = IndexedGraph.FNode(
                idx, Pattern.get(graph.ops[nid].type), list(), nid)
            self.topo_order.append(self.node_map[nid])
        for nid, prev_set in redges.items():
            for prev_nid in prev_set:
                self.node_map[prev_nid].outputs.append(
                    IndexedGraph.FEdge(self.node_map[nid],
                                       self.node_map[nid].pattern))

        # for nid in self.node_map:
        #     print(graph.ops[nid], self.node_map[nid])
        # print('>>>>>>')


class DominatorTree:
    class TNode:
        def __init__(self,
                     gnode=None,
                     parent=None,
                     depth=0,
                     pattern=Pattern.Compbound):
            self.gnode = gnode
            self.parent = parent
            self.depth = depth
            self.pattern = pattern

    def __init__(self, num_nodes):
        self.nodes = [None] * num_nodes

    def combine_pattern(self, lhs, rhs):
        return max(lhs, rhs)

    def LCA(self, lhs, rhs, edge_pattern):
        while lhs != rhs:
            if lhs is None or rhs is None:
                return None
            if lhs.depth < rhs.depth:
                edge_pattern = self.combine_pattern(edge_pattern, rhs.pattern)
                rhs = rhs.parent
            elif rhs.depth < lhs.depth:
                edge_pattern = self.combine_pattern(edge_pattern, lhs.pattern)
                lhs = lhs.parent
            else:
                edge_pattern = self.combine_pattern(edge_pattern, rhs.pattern)
                edge_pattern = self.combine_pattern(edge_pattern, lhs.pattern)
                rhs = rhs.parent
                lhs = lhs.parent
        return lhs

    def least_common_ancestor(self, in_nodes, edge_pattern):
        if len(in_nodes) == 0:
            return None, edge_pattern

        def get_dnode(edge: IndexedGraph.FEdge):
            oindex = edge.node.index
            return self.nodes[oindex]

        edges = list(in_nodes)
        parent = get_dnode(edges[0])
        edge_pattern = self.combine_pattern(edge_pattern, edges[0].pattern)
        for e in edges[1:]:
            parent = self.LCA(parent, get_dnode(e), edge_pattern)
            edge_pattern = self.combine_pattern(edge_pattern, e.pattern)
        return parent, edge_pattern

    def get_node(self, gnode: IndexedGraph.FNode):
        tnode = DominatorTree.TNode()
        tnode.gnode = gnode
        parent, pattern = self.least_common_ancestor(gnode.outputs,
                                                     Pattern.Membound)
        tnode.depth = 1 if parent is None else parent.depth + 1
        tnode.parent = parent
        tnode.pattern = pattern
        return tnode

    @staticmethod
    def post_dom(graph: IndexedGraph):
        tree = DominatorTree(len(graph.topo_order))
        for i in range(len(graph.topo_order) - 1, -1, -1):
            tree.nodes[i] = tree.get_node(graph.topo_order[i])
        return tree


class GraphPartitioner:
    class Group:
        _id = 0

        # union find
        def __init__(self,
                     group,
                     pattern,
                     root_ref=None,
                     anchor_ref=None,
                     num_nodes=1):
            self.parent = group
            self.pattern = pattern
            self.root_ref = root_ref
            self.anchor_ref = anchor_ref
            self.num_nodes = num_nodes

            self.id = GraphPartitioner.Group._id
            GraphPartitioner.Group._id += 1

        def find_root(self):
            if self.parent is None:
                return self
            root = self
            while root.parent is not None:
                root = root.parent

            p = self
            while p != root:
                parent = p.parent
                p.parent = root
                p = parent
            return root

    def __init__(self, max_fuse_depth=1000):
        self.max_fuse_depth = max_fuse_depth
        self._groups = []
        self._visited = set()

    def check_path_(self, src: IndexedGraph.FNode, sink: IndexedGraph.FNode,
                    fcond):
        if src.index in self._visited:
            return True
        self._visited.add(src.index)
        gnode = self._groups[src.index].find_root()
        if not fcond(gnode.pattern, src == sink):
            return False
        if src == sink:
            return True
        for e in src.outputs:
            if not self.check_path_(e.node, sink, fcond):
                return False
        return True

    def check_path(self, src: IndexedGraph.FNode, sink: IndexedGraph.FNode,
                   fcond):
        self._visited.clear()
        for e in src.outputs:
            if not self.check_path_(e.node, sink, fcond):
                return False
        return True

    def combine_pattern(self, lhs, rhs):
        return max(lhs, rhs)

    def merge_from_to(self, child, parent):
        child = child.find_root()
        parent = parent.find_root()
        if child == parent:
            return
        parent.num_nodes += child.num_nodes
        child.parent = parent
        if child.anchor_ref is not None:
            parent.anchor_ref = child.anchor_ref
            parent.pattern = self.combine_pattern(child.pattern, parent.pattern)

    def commit_fuse_(self, src: IndexedGraph.FNode, sink: IndexedGraph.FNode,
                     target):
        if src == sink or src.index in self._visited:
            return
        self._visited.add(src.index)
        gnode = self._groups[src.index]
        self.merge_from_to(gnode, target)
        for e in src.outputs:
            self.commit_fuse_(e.node, sink, target)

    def commit_fuse(self, src: IndexedGraph.FNode, sink: IndexedGraph.FNode):
        target = self._groups[sink.index]
        self._visited.clear()
        assert src != sink
        self.commit_fuse_(src, sink, target)

    def init_groups(self, graph: IndexedGraph):
        for nid in range(len(graph.topo_order)):
            graph_node = graph.topo_order[nid]
            group_node = GraphPartitioner.Group(None,
                                                graph_node.pattern,
                                                root_ref=graph_node)
            if group_node.pattern == Pattern.Compbound:
                group_node.anchor_ref = graph_node
            self._groups.append(group_node)

    def run_fuse(self, graph: IndexedGraph, post_dom_tree: DominatorTree,
                 phase: int):
        for nid in range(len(self._groups)):
            graph_node = graph.topo_order[nid]
            dom_node = post_dom_tree.nodes[nid]
            group_node = self._groups[nid]

            if group_node.pattern == Pattern.Opaque:
                continue
            if dom_node.parent is None:
                continue
            dom_parent_gindex = dom_node.parent.gnode.index

            # self.count_fused_nodes_with_new_child(graph_node, dom_node.parent.gnode)

            # if phase == 2:

            if self._groups[
                    dom_parent_gindex] is not None and group_node.find_root(
                    ) == self._groups[dom_parent_gindex].find_root():
                continue
            if group_node.pattern == Pattern.Compbound:
                if phase != 0:
                    continue
                if dom_node.parent is not None and dom_node.pattern == Pattern.Membound:
                    fcond = lambda kind, is_sink: kind <= Pattern.Membound
                    if self.check_path(graph_node, dom_node.parent.gnode,
                                       fcond):
                        self.commit_fuse(graph_node, dom_node.parent.gnode)
            # elif

    def partition(self, graph: IndexedGraph):
        self.init_groups(graph)
        post_dom_tree = DominatorTree.post_dom(graph)
        for phase in range(1):
            self.run_fuse(graph, post_dom_tree, phase)
        return self._groups


class FuseMutator:
    def __init__(self):
        self.gmap = defaultdict(list)
        self.anchor = defaultdict(set)

    def transform(self,
                  pgraph: ProteusModel,
                  max_fuse_depth=1000,
                  forward=False):
        index_graph = IndexedGraph(pgraph, forward=True)
        groups = GraphPartitioner(
            max_fuse_depth=max_fuse_depth).partition(index_graph)

        group_map = {}
        for idx in range(len(index_graph.topo_order)):
            group_map[index_graph.topo_order[idx].orgid] = groups[idx]

            group_id = groups[idx].find_root().id
            self.gmap[group_id].append(index_graph.topo_order[idx].orgid)
            if groups[idx].anchor_ref is not None:
                self.anchor[group_id].add(groups[idx].anchor_ref.orgid)

        if not forward:
            iter_gmap = self.gmap.copy()
            for gid, group in iter_gmap.items():
                ngid = GraphPartitioner.Group._id + gid
                for fid in group:
                    if fid in pgraph.fwop_map:
                        self.gmap[ngid].append(pgraph.fwop_map[fid])
                if gid in self.anchor:
                    for fid in self.anchor[gid]:
                        self.anchor[ngid].add(pgraph.fwop_map[fid])
        return self

    def rewrite_graph(self, pgraph: ProteusModel):
        for gid, group in self.gmap.items():
            if len(group) > 1:
                FusedOp.create(pgraph, group, self.anchor[gid])
        return pgraph


def run_fuse(graph, forward=False):
    mutator = FuseMutator().transform(graph, forward=forward)
    graph = mutator.rewrite_graph(graph)
    return graph
