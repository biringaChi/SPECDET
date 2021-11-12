import torch
import torch.nn as nn

class SelfAttention(nn.Module):
	def __init__(self, emebedding_size, heads) -> None:
		super(SelfAttention, self).__init__()
		self.heads = heads
		self.embedding_size = emebedding_size
		self.head_dimension = emebedding_size // heads
		assert(self.head_dimension * heads == emebedding_size), f"Invalid {self.embedding_size} must be divisible by {self.heads}"
		self.keys = nn.Linear(self.head_dimension, self.head_dimension, bias = False)
		self.values = nn.Linear(self.head_dimension, self.head_dimension, bias = False)
		self.queries = nn.Linear(self.head_dimension, self.head_dimension, bias = False)
		self.fc_out = nn.Linear(heads * self.head_dimension, emebedding_size)
	
	def __str__(self) -> str:
		return f"{self.__class__.__name__}"

	def __repr__(self) -> str:
		return self.__str__()

	def forward(self, keys, values, query, mask):
		observation_size = query.shape[0]
		value_length, key_length, query_length = values.shape[1], keys.shape[1], query.shape[1]

		values = values.reshape(observation_size, value_length, self.heads, self.head_dimension)
		keys = keys.reshape(observation_size, key_length, self.heads, self.head_dimension)
		queries = query.reshape(observation_size, query_length, self.heads, self.head_dimension)

		values = self.values(values)
		keys = self.keys(keys)
		queries = self.queries(queries)

		energy = torch.einsum("oqhd,okhd->ohqk", [queries, keys])
		if mask is not None:
			energy = energy.masked_fill(mask == 0, float("-1e20"))

		attention = torch.softmax(energy / (self.embedding_size ** (1/2)), dim = 3)
		out = torch.einsum("ohql,olhd->oqhd", [attention, values]).reshape(observation_size, query_length, self.heads * self.head_dimension)
		return self.fc_out(out)
