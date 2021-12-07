from typing import Tuple
import torch
import time
import numpy as np
import torch.nn as nn
from data_prep import DataSample60K
import torch.optim as optim
from data_prep import Embedding
from torch.utils.data import sampler
from torch.utils.data import DataLoader
from model import CNN

net = CNN()
torch.manual_seed(60)


class Train(Embedding):
	"""
	Trains the Convolutional Neural Network (CNN)
	Args: 
		train: A tensor representing training samples of benign and spectre assembly code
		test: A tensor representing testing samples of benign and spectre assembly code
	"""
	def __init__(self) -> None:
		super().__init__()
		self.CRITERION = nn.BCELoss()
		self.OPTIMIZER = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
		self.EPOCHS = 10
		self.VALIDATION_LOSS_MIN = np.inf
		self.TRAIN_LOSSES, self.TRAIN_ACCURACIES, self.VALIDATION_LOSSES, self.VALIDATION_ACCURACIES = ([] for _ in range(4))
		self.EPOCH_TRAIN_LOSSES, self.EPOCH_TRAIN_ACCURACIES, self.EPOCH_VALIDATION_LOSSES, self.EPOCH_VALIDATION_ACCURACIES = ([] for _ in range(4))

	def training_targets(self):
		return torch.Tensor(self.benign_spectre_train_targets())

	def data_loader(self) -> Tuple:
		# get loader train and val
		loader_train = DataLoader(torch.nn.functional.normalize(torch.Tensor(self.training()), batch_size=64, sampler=sampler.SubsetRandomSampler(range(self.NUM_TRAIN))))
		loader_val = DataLoader(torch.nn.functional.normalize(torch.Tensor(self.training()), batch_size=64, sampler=sampler.SubsetRandomSampler(range(self.NUM_TRAIN, self.VALIDATION))))
		loader_test = DataLoader(torch.nn.functional.normalize(torch.Tensor(self.testing()), batch_size=64))
		return loader_train, loader_val, loader_test
	
	def accuracy(self):
		cnn = CNN()
		cnn.eval()
		correct = 0
		total = 0
		_, _, loader_test = self.dataloader()
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