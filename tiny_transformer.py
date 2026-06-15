import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.shape
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            if mask.dim() == 3: mask = mask.unsqueeze(1)
            elif mask.dim() == 2: mask = mask.unsqueeze(0).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attention_weights, v)
        concat_out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_linear(concat_out)


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

    def compute_prefix_lm_mask(self, input_ids: torch.Tensor, eq_token_id: int) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device))
        
        eq_indices = (input_ids == eq_token_id).int().argmax(dim=-1)
        eq_grid = eq_indices.view(batch_size, 1, 1)
        col_grid = torch.arange(seq_len, device=input_ids.device).view(1, 1, seq_len)
        
        is_prefix_col = (col_grid <= eq_grid)
        prefix_lm_mask = torch.where(is_prefix_col, 1.0, causal_mask)
        
        return prefix_lm_mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.token_embedding_layer(x)
        x = self.positional_encoding(x)
        
        attn_residual = x
        x_norm1 = self.pre_attention_layernorm(x)
        attn_out = self.multi_head_attention(x_norm1, mask=mask)
        x = attn_residual + attn_out
        
        ffn_residual = x
        x_norm2 = self.pre_ffn_layernorm(x)
        ffn_out = self.ffn(x_norm2)
        x = ffn_residual + ffn_out
        
        logits = self.output_linear(x)
        
        return logits

