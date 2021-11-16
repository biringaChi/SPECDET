import torch
import torch.nn as nn

from attention import SelfAttention

class TransformerBlock(nn.Module):
	def __init__(self, emebedding_size, heads, dropout, forward_expansion) -> None:
		super().__init__(TransformerBlock, self).__init__()
		self.attention = SelfAttention(emebedding_size, heads)
		self.layer_norm1 = nn.LayerNorm(emebedding_size)
		self.layer_norm2 = nn.LayerNorm(emebedding_size)
		self.feed_forward = nn.Sequential(
			nn.Linear(emebedding_size, forward_expansion * emebedding_size),
			nn.ReLU(),
			nn.Linear(forward_expansion * emebedding_size, emebedding_size)
			)
		self.dropout = nn.Dropout(dropout)

	def __str__(self) -> str:
		return f"{self.__class__.__name__}"

	def __repr__(self) -> str:
		return self.__str__()

	def forward(self, value, key, query, mask):
		attention = self.attention(value, key, query, mask)
		x = self.dropout(self.layer_norm1(attention + query))
		forward = self.feed_forward(x)
		out = self.dropout(self.layer_norm2(forward + x))
		return out
		