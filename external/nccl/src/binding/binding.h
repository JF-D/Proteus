#ifndef NCCL_BINDING_H_
#define NCCL_BINDING_H_

#include <vector>
#include <string>

#include "comm.h"
#include "nccl.h"
#include "xml.h"
#include "topo.h"
#include "enqueue.h"
#include "transport.h"

class Communicator {
public:
  Communicator(std::vector<std::vector<int>> groups, std::vector<std::vector<int>> groupsRank,
      int nRanks, int nNodes, std::string topofile);

  std::vector<float> broadcast(size_t nBytes, int root);

  std::vector<float> reduce(size_t nBytes, int root);

  std::vector<float> allreduce(size_t nBytes);

  std::vector<float> allgather(size_t nBytes);

  std::vector<float> reducescatter(size_t nBytes);

  int get_graph_type_intra();

  int get_graph_type_inter();

  bool get_cross_node();

  struct ncclComm* comm;
  int typeIntra_, typeInter_;
  bool crossNode_;
};


#endif
