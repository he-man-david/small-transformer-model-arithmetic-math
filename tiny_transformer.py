import math
import torch
import torch.nn as nn

# An unoptimized vanilla MHA - add the new shit in future
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear_heads = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.k_linear_heads = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.v_linear_heads = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])

        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = []
        
        for i in range(self.num_heads):
            q_i = self.q_linear_heads[i](x)
            k_i = self.k_linear_heads[i](x)
            v_i = self.v_linear_heads[i](x)
            
            scores_i = torch.matmul(q_i, k_i.transpose(-2, -1))
            scores_i = scores_i / (self.d_k ** 0.5)
            
            attention_weights_i = torch.softmax(scores_i, dim=-1)
            attention_weights_i = self.dropout(attention_weights_i)
            
            out_i = torch.matmul(attention_weights_i, v_i)
            head_outputs.append(out_i)
            
        concat_out = torch.cat(head_outputs, dim=-1)
        final_output = self.out_linear(concat_out)
        
        return final_output


# The OG positional encoding from 2017 paper - I will try RoPE in future
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TinyArithmeticTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, max_seq_len: int = 100, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.token_embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.pre_attention_layernorm = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.pre_ffn_layernorm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),                         
            nn.Linear(4 * d_model, 4 * d_model),
            nn.ReLU(),                         
            nn.Linear(4 * d_model, d_model)
        )
        
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding_layer(x)
        x = self.positional_encoding(x)
        
        attn_residual = x
        x_norm1 = self.pre_attention_layernorm(x)
        attn_out = self.multi_head_attention(x_norm1)
        x = attn_residual + attn_out
        
        ffn_residual = x
        x_norm2 = self.pre_ffn_layernorm(x)
        ffn_out = self.ffn(x_norm2)
        x = ffn_residual + ffn_out
        
        logits = self.output_linear(x)
        
        return logits

