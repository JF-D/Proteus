import proteus.torchapi as torch
import proteus.torchapi.nn as nn
import proteus.torchapi.nn.functional as F


class MLP(nn.Module):

    def __init__(self, hidden_size, output_dropout_prob, bias=True):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.activation = nn.Activation('gelu')
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(p=output_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class Attention(nn.Module):

    def __init__(self, hidden_size, nheads, attention_dropout_prob,
                 output_dropout_prob, seq_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.nheads = nheads
        self.hidden_size_per_head = hidden_size // nheads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)

        self.scaled_attention_mask = nn.Elementwise('attention_mask')

        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(p=attention_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = nn.Dropout(p=output_dropout_prob)

        self.seq_first = seq_first

    def forward(self, hidden_states, attention_mask):
        if self.seq_first:
            # [sq, b, h] --> [sq, b, 3h]
            mixed_x_layer = self.qkv(hidden_states)
            # [sq, b, 3h] --> [sq, b, n, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                    (self.nheads, 3 * self.hidden_size_per_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            # [sq, b, n, 3 * hn] --> 3 [sq, b, n, hn]
            (query_layer, key_layer,
             value_layer) = torch.split(mixed_x_layer,
                                        self.hidden_size_per_head,
                                        dim=-1)
            # [sq, b, n, hn] --> [sq, b * n, hn]
            query_layer = query_layer.view(
                query_layer.size(0),
                query_layer.size(1) * query_layer.size(2), query_layer.size(3))
            # [sq, b, n, hn] --> [sq, b * n, hn]
            key_layer = key_layer.view(key_layer.size(0),
                                       key_layer.size(1) * key_layer.size(2),
                                       key_layer.size(3))
            # attention score: [b * n, sq, sq]
            attention_score = torch.matmul(query_layer.permute(1, 0, 2),
                                           key_layer.permute(1, 2, 0))
            # [b * n, sq, sq] --> [b, n, sq, sq]
            attention_score = attention_score.view(-1, self.nheads,
                                                   attention_score.size(1),
                                                   attention_score.size(2))
            # >>> score in one op
            attention_score = self.scaled_attention_mask(attention_score,
                                                         attention_mask)
            attention_probs = self.softmax(attention_score)
            attention_probs = self.attention_dropout(attention_probs)

            # [sq, b, n, hn] --> [sq, b * n, hn]
            value_layer = value_layer.view(
                value_layer.size(0),
                value_layer.size(1) * value_layer.size(2), value_layer.size(3))
            # [b, n, sq, sq] --> [b * n, sq, sq]
            attention_probs = attention_probs.view(-1, attention_probs.size(2),
                                                   attention_probs.size(3))
            # context layer: [b * n, sq, hn]
            context_layer = torch.matmul(attention_probs,
                                         value_layer.permute(1, 0, 2))
            # [b * n, sq, hn] --> [b, n, sq, hn]
            context_layer = context_layer.view(-1, self.nheads,
                                               context_layer.size(1),
                                               context_layer.size(2))
            # [b, n, sq, hn] --> [sq, b, n, hn]
            context_layer = context_layer.permute(2, 0, 1, 3)
            # [sq, b, n, hn] --> [sq, b, h]
            new_tensor_shape = context_layer.size()[:-2] + (self.hidden_size, )
            context_layer = context_layer.reshape(new_tensor_shape)
        else:
            # [b, sq, h] --> [b, sq, 3h]
            mixed_x_layer = self.qkv(hidden_states)
            # [b, sq, 3h] --> [b, sq, n, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                    (self.nheads, 3 * self.hidden_size_per_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            # [b, sq, n, 3 * hn] --> [b, n, sq, 3 * hn]
            mixed_x_layer = mixed_x_layer.permute(0, 2, 1, 3)
            # [b, n, sq, 3 * hn] --> 3 [b, n, sq, hn]
            (query_layer, key_layer,
             value_layer) = torch.split(mixed_x_layer,
                                        self.hidden_size_per_head,
                                        dim=-1)
            # [b, n, sq, hn] --> [b, n, hn, sq]
            key_layer = key_layer.permute(0, 1, 3, 2)
            # attention score: [b, n, sq, sq]
            attention_score = torch.matmul(query_layer, key_layer)
            # >>> score in one op
            attention_score = self.scaled_attention_mask(attention_score,
                                                         attention_mask)
            attention_probs = self.softmax(attention_score)
            attention_probs = self.attention_dropout(attention_probs)

            # context layer: [b, n, sq, hn]
            context_layer = torch.matmul(attention_probs, value_layer)
            # [b, n, sq, hn] --> [b, sq, n, hn]
            context_layer = context_layer.permute(0, 2, 1, 3)
            # [b, sq, n, hn] --> [b, sq, h]
            new_tensor_shape = context_layer.size()[:-2] + (self.hidden_size, )
            context_layer = context_layer.reshape(new_tensor_shape)

        out = self.dense(context_layer)
        out = self.output_dropout(out)
        return out


class GPTLayer(nn.Module):

    def __init__(self, hidden_size, nheads, attention_dropout_prob,
                 output_dropout_prob, seq_first=True):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.attention = Attention(hidden_size, nheads, attention_dropout_prob,
                                   output_dropout_prob, seq_first=seq_first)
        self.attention_add = nn.Elementwise('add')
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, output_dropout_prob)
        self.mlp_add = nn.Elementwise('add')

    def forward(self, hidden_states, attention_mask):
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output = self.attention(layernorm_output, attention_mask)
        layernorm_input = self.attention_add(hidden_states, attention_output)
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        mlp_output = self.mlp(layernorm_output)
        output = self.mlp_add(layernorm_input, mlp_output)
        return output


class Embedding(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_seq_length,
                 hidden_size,
                 output_dropout_prob=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.add = nn.Elementwise('add')
        self.dropout = nn.Dropout(output_dropout_prob)

    def forward(self, x, position_ids):
        we = self.word_embeddings(x)
        pe = self.position_embeddings(position_ids)
        out = self.add(we, pe)
        out = self.dropout(out)
        return out


class GPT(nn.Module):

    def __init__(self,
                 num_layers,
                 max_seq_length,
                 hidden_size,
                 nheads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 vocab_size,
                 seq_first=True):
        super().__init__()
        self.seq_first = seq_first
        self.embedding = Embedding(vocab_size, max_seq_length, hidden_size)
        self.layers = torch.nn.ModuleList([
            GPTLayer(hidden_size,
                     nheads,
                     attention_dropout_prob,
                     output_dropout_prob,
                     seq_first=self.seq_first) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x, position_ids, mask):
        x = self.embedding(x, position_ids)
        if self.seq_first:
            x = x.permute(1, 0, 2)

        for i in range(self.num_layers):
            x = self.layers[i](x, mask)

        if self.seq_first:
            x = x.permute(1, 0, 2)
        x = F.linear(x, self.embedding.word_embeddings.weight)
        return x
