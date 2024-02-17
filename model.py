import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attention_projection = nn.Linear(
            config.embedding_dimension, 3 * config.embedding_dimension, bias=config.bias
        )
        self.output_projection = nn.Linear(
            config.embedding_dimension, config.embedding_dimension, bias=config.bias
        )

        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        self.n_heads = config.n_heads
        self.embedding_dimension = config.embedding_dimension
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.attention_projection(x).split(self.embedding_dimension, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.output_projection(y)
        y = self.residual_dropout(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.linear1 = nn.Linear(
            config.embedding_dimension, 4 * config.embedding_dimension, bias=config.bias
        )
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(
            4 * config.embedding_dimension, config.embedding_dimension, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.embedding_dimension)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dimension)
        self.masked_multihead_attention = CausalSelfAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        layer_norm1_output = self.layer_norm1(x)
        masked_multihead_attention_output = self.masked_multihead_attention(
            layer_norm1_output
        )
        x = x + masked_multihead_attention_output
        layer_norm2_output = self.layer_norm2(x)
        feed_forward_output = self.feed_forward(layer_norm2_output)
        x = x + feed_forward_output
        return x


class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.token_embedder = nn.Embedding(
            config.vocab_size, config.embedding_dimension
        )
        self.positional_embedder = nn.Embedding(
            config.sequence_length, config.embedding_dimension
        )
        self.dropout = nn.Dropout(config.dropout)
        self.decoder_layers = nn.ModuleList(
            [Decoder(config) for _ in range(config.n_layers)]
        )
        self.layer_normalizar = nn.LayerNorm(config.embedding_dimension)
        self.language_model_head = nn.Linear(
            config.embedding_dimension, config.vocab_size, bias=False
        )

    def forward(self, idx, targets=None):
        b, t = idx.size()
        position = torch.arange(0, t, dtype=torch.long, device=self.device)
        # token and positional embedding
        token_embedding = self.token_embedder(idx)
        positional_embedding = self.positional_embedder(position).to(self.device)
        x = self.dropout(token_embedding + positional_embedding)
        # decoder layers
        for decoder in self.decoder_layers:
            x = decoder(x)
        x = self.layer_normalizar(x)

        if targets is not None:
            logits = self.language_model_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.language_model_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)  # this goes back to the forward function
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes batch, channel
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # batch, channel
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # batch,1
            # append samples index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # batch, time +1
        return idx
