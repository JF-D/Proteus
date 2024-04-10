import proteus.torchapi as torch
import proteus.torchapi.nn as nn


class DLRM(nn.Module):
    def __init__(self, m_spa, ln_emb, ln_bot, ln_top, arch_interaction_op='dot',
                 arch_interaction_itself=False, sigmoid_bot=-1, sigmoid_top=-1,
                 num_indices_per_lookup=100, divide=False):
        super().__init__()
        self.arch_interaction_op = arch_interaction_op
        self.arch_interaction_itself = arch_interaction_itself
        self.num_indices_per_lookup = num_indices_per_lookup
        self.divide = divide

        self.emb_l = self.create_emb(m_spa, ln_emb)
        self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
        self.top_l = self.create_mlp(ln_top, sigmoid_top)

    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        for i in range(0, ln.size):
            n = ln[i]
            EE = nn.Embedding(n, m, attr=dict(type='bag', mode='sum', num_indices_per_lookup=self.num_indices_per_lookup))
            emb_l.append(EE)
        return emb_l

    def create_mlp(self, ln, sigmoid_layer):
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n, m = ln[i], ln[i + 1]
            LL = nn.Linear(int(n), int(m), bias=True)
            layers.append(LL)

            if i == sigmoid_layer:
                layers.append(nn.Activation('sigmoid'))
            else:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def interact_features(self, x, ly):
        if self.arch_interaction_op == 'dot':
            (batch_size, d) = x.shape
            if self.divide:
                T1 = torch.cat([x] + ly[:16], dim=1)
                T2 = torch.cat(ly[16:], dim=1)
                T = torch.cat([T1, T2], dim=1).view((batch_size, -1, d))
            else:
                T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            Z = torch.matmul(T, T.permute(0, 2, 1))

            _, ni, nj = Z.shape
            offset = 1 if self.arch_interaction_itself else 0
            Z = Z.reshape((batch_size, -1))
            Zflat = Z.slice(len([i for i in range(ni) for j in range(i + offset)]))
            R = torch.cat([x, Zflat], dim=1)
        else:
            assert False, f'Unknow interaction op {self.arch_interaction_op}!'

        return R

    def forward(self, dense_X, *sparse_X):
        # apply emb
        ly = []
        for k, spar_x in enumerate(sparse_X):
            V = self.emb_l[k](spar_x)
            ly.append(V)

        # apply bot mlp
        x = self.bot_l(dense_X)

        # interaction
        z = self.interact_features(x, ly)

        # top mlp
        p = self.top_l(z)
        return p
