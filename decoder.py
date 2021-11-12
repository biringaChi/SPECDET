import torch
import torch.nn as nn

from attention import SelfAttention
from tb import TransformerBlock

class DecoderBlock(nn.Module):
	def __init__(
		self, 
		emebedding_size, 
		heads, 
		dropout, 
		forward_expansion, 
		device
		) -> None:
		super(DecoderBlock, self).__init__()
		self.attention = SelfAttention(emebedding_size, heads)
		self.layer_norm = nn.LayerNorm(emebedding_size)
		self.transformer_block = TransformerBlock(emebedding_size, heads, dropout, forward_expansion)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x, value, key, source_mask, target_mask):
		attention = self.attention(x, x, x, target_mask)
		query = self.dropout(self.layer_norm(attention + x))
		out = self.transformer_block(value, key, query, source_mask)
		return out

class Decoder(nn.Module):
	def __init__(
		self, 
		target_vocabulary_size, 
		emebedding_size, 
		heads, 
		number_layers, 
		forward_expansion, 
		dropout, 
		device, 
		maximum_length
		) -> None:
		super(Decoder, self).__init__()
		self.device = device
		self.word_embedding = nn.Embedding(target_vocabulary_size, emebedding_size)
		self.position_embedding = nn.Embedding(maximum_length, emebedding_size)
		self.layers = nn.ModuleList(
			[DecoderBlock(emebedding_size, heads, forward_expansion, dropout, device) for _ in range(number_layers)]
		)
		self.fc_out = nn.Linear(emebedding_size, target_vocabulary_size)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x, encoder_out, source_mask, target_mask):
		observation_size, sequence_length = x.shape
		positions = torch.arange(0, sequence_length).expand(observation_size, sequence_length).to(self.device)
		x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
		for layer in self.layers:
			x = layer(x, encoder_out, encoder_out, source_mask, target_mask)
		out = self.fc_out(x)
		return out
