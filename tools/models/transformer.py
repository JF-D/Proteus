import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


# class LayerNorm(nn.Module):

#     def __init__(self, hidden_size, eps=1e-12):
#         """Construct a layernorm module in the TF style (epsilon inside the square root).
#         """
#         super(LayerNorm, self).__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.bias = nn.Parameter(torch.zeros(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.variance_epsilon)
#         return self.weight * x + self.bias
LayerNorm = torch.nn.LayerNorm

class MLP(nn.Module):

    def __init__(self, hidden_size, output_dropout_prob, bias=True):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.activation = gelu
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
                 output_dropout_prob):
        super().__init__()
        self.hidden_size = hidden_size
        self.nheads = nheads
        self.hidden_size_per_head = hidden_size // nheads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(p=attention_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = nn.Dropout(p=output_dropout_prob)

    def forward(self, hidden_states, attention_mask):
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
        attention_score = attention_score / math.sqrt(self.hidden_size_per_head)
        attention_score = torch.mul(
            attention_score, attention_mask) - 10000.0 * (1.0 - attention_mask)
        attention_probs = self.softmax(attention_score)
        attention_probs = self.attention_dropout(attention_probs)

        # context layer: [b, n, sq, hn]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, n, sq, hn] --> [b, sq, n, hn]
        context_layer = context_layer.permute(0, 2, 1, 3)
        # [b, sq, n, hn] --> [b, sq, h]
        new_tensor_shape = context_layer.size()[:-2] + (self.hidden_size, )
        context_layer = context_layer.reshape(*new_tensor_shape)

        out = self.dense(context_layer)
        out = self.output_dropout(out)
        return out


class GPTLayer(nn.Module):

    def __init__(self, hidden_size, nheads, attention_dropout_prob,
                 output_dropout_prob):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        # self.input_layernorm = LayerNorm(hidden_size)
        self.attention = Attention(hidden_size, nheads, attention_dropout_prob,
                                   output_dropout_prob)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        # self.post_attention_layernorm = LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, output_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output = self.attention(layernorm_output, attention_mask)
        layernorm_input = hidden_states + attention_output
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        mlp_output = self.mlp(layernorm_output)
        output = layernorm_input + mlp_output
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
        self.dropout = nn.Dropout(output_dropout_prob)

    def forward(self, x, position_ids):
        we = self.word_embeddings(x)
        pe = self.position_embeddings(position_ids)
        out = we + pe
        out = self.dropout(out)
        return out


class GPT(nn.Module):

    def __init__(self, num_layers, max_seq_length, hidden_size, nheads,
                 attention_dropout_prob, output_dropout_prob, vocab_size):
        super().__init__()
        self.embedding = Embedding(vocab_size, max_seq_length, hidden_size,
                                   output_dropout_prob)
        self.layers = torch.nn.ModuleList([
            GPTLayer(hidden_size, nheads, attention_dropout_prob,
                     output_dropout_prob) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x, position_ids, mask):
        x = self.embedding(x, position_ids)
        for i in range(self.num_layers):
            x = self.layers[i](x, mask)
        x = F.linear(x, self.embedding.word_embeddings.weight)
        return x
