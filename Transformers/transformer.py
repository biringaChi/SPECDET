import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
	def __init__(
		self, 
		source_vocabulary_size, 
		target_vocabulary_size, 
		source_pad_idx, 
		target_pad_idx, 
		embedding_size = 256, 
		number_layers = 6,
		forward_expansion = 4,
		heads = 8, 
		dropout = 0,
		device = "cuda",
		maximum_length = 100 
		) -> None:
		super(Transformer, self).__init__()

		self.encoder = Encoder(
			source_vocabulary_size,
			embedding_size,
			number_layers,
			heads,
			device,
			forward_expansion, 
			dropout, 
			maximum_length
		)
		self.decoder = Decoder(
			target_vocabulary_size,
			embedding_size, 
			number_layers, 
			heads, 
			forward_expansion,
			dropout,
			device, 
			maximum_length 
		)
		
		self.source_pad_idx = source_pad_idx
		self.target_pad_idx = target_pad_idx
		self.device = device

	def __str__(self) -> str:
		return f"{self.__class__.__name__}"

	def __repr__(self) -> str:
		return self.__str__()
	
	def make_source_mask(self, source):
		source_mask = (source != self.source_pad_idx).unsqueeze(1).unsqueeze(2)
		return source_mask.to(self.device)
	
	def make_target_mask(self, target):
		observation_size, target_length = target.shape
		target_mask = torch.tril(torch.ones((target_length, target_length))).expand(
			observation_size, 1, target_length, target_length
		)
		return target_mask.to(self.device)

	def forward(self, source, target):
		source_mask = self.make_source_mask(source)
		target_mask = self.make_source_mask(target)
		encode_source = self.encoder(source, source_mask)
		out = self.decoder(target, encode_source, source_mask, target_mask)
		return out
