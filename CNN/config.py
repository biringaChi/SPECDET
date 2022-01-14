from typing import List, Tuple

class Config:
	"""
	SpectreCNN configuration arguments
	"""
	
	def __init__(
				 self, 
				 epochs: int = 2, 
				 learning_rate: float = 0.001, 
				 betas: Tuple[float] = (0.5, 0.999),
				 label_num: int = 1, 
				 in_channels: int = 1,
				 out_channels: int = 1, 
				 in_features: int = 10, 
				 out_features: int = 300,
				 drop_out_prob: float = 0.5,
				 kernel_size: int = 3, 
				 stride: int = 1, 
				 padding: List[Tuple] = [(1, 0), (2, 0), (3, 0)], 
				 batch_size: int = 32, 
				 max_pool = (10, 10)
				 ) -> None:
		self.EPOCHS = epochs
		self.LEARNING_RATE = learning_rate
		self.BETAS = betas
		self.LABEL_NUM = label_num
		self.IN_CHANNELS = in_channels 
		self.IN_FEATURES = in_features 
		self.OUT_FEATURES = out_features
		self.DROP_OUT_PROB = drop_out_prob
		self.OUT_CHANNELS = out_channels
		self.KERNEL_SIZE = kernel_size
		self.STRIDE = stride
		self.PADDING = padding
		self.BATCH_SIZE = batch_size
		self.MAX_POOL = max_pool