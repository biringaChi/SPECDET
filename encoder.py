import torch
import torch.nn as nn
from torch.nn.modules import dropout

from tb import TransformerBlock

class Encoder(nn.Module):
	def __init__(
		self, 
		source_vocabulary_size, 
		emebedding_size, 
		number_layers,
		heads, 
		device, 
		forward_expansion, 
		maximum_length
		) -> None:
		super(Encoder, self).__init__()
		self.emebedding_size = emebedding_size
		self.device = device
		self.word_embedding = nn.Embedding(source_vocabulary_size, emebedding_size)
		self.position_embedding = nn.Embedding(maximum_length, emebedding_size)

		self.layers = nn.ModuleList(
			[TransformerBlock(emebedding_size, heads, dropout = dropout, forward_expansion = forward_expansion) for _ in range(number_layers)]
		)
		self.dropout = nn.Dropout(dropout)
	
	def __str__(self) -> str:
		return f"{self.__class__.__name__}"

	def __repr__(self) -> str:
		return self.__str__()
	
	def forward(self, x, mask):
		observation_size, sequence_length = x.shape
		positions = torch.arange(0, sequence_length).expand(observation_size, sequence_length).to(self.device)
		out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
		for layer in self.layers:
			out = layer(out, out, out, mask)
		return out
		