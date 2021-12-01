from typing import Tuple
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data_prep import Embedding
from torch.utils.data import sampler
from torch.utils.data import DataLoader
from model import CNN

net = CNN()
torch.manual_seed(60)

class Train(Embedding):
	def __init__(self) -> None:
		super().__init__()
		self.NUM_TRAIN = 938_519 
		self.VALIDATION = 957_673
		self.CRITERION = nn.CrossEntropyLoss()
		self.OPTIMIZER = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
		self.EPOCHS = 10
		self.VALIDATION_LOSS_MIN = np.inf

	def training_targets(self):
		return torch.Tensor(self.benign_spectre_train_targets())

	def data_loader(self) -> Tuple:
		loader_train = DataLoader(torch.Tensor(self.training()), batch_size=64, sampler=sampler.SubsetRandomSampler(range(self.NUM_TRAIN)))
		loader_val = DataLoader(torch.Tensor(self.training()), batch_size=64, sampler=sampler.SubsetRandomSampler(range(self.NUM_TRAIN, 957_673)))
		loader_test = DataLoader(torch.Tensor(self.testing()), batch_size=64)
		return loader_train, loader_val, loader_test
	
	def accuracy(self):
		cnn = CNN()
		cnn.eval()
		correct = 0
		total = 0
		loader_train, loader_val, loader_test = self.dataloader()
		for data in loader_test:
			benign_spectre_train, labels = data
			outputs = net(benign_spectre_train)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			acc = 100 * (correct/total)
		return acc

	def main(self):
		pass


if __name__ == "__main__":
	pass