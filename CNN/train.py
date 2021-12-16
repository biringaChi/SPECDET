import os
import torch
import time
from tqdm import tqdm
import numpy as np
from cnn import CNN
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from torch.functional import Tensor
from data_prep import SpectreEmbedding
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(60)

class TrainTest(SpectreEmbedding):
	"""
	Trains the Convolutional Neural Network (CNN)
	Args: 
		train: A tensor representing training samples of benign and spectre assembly code
		test: A tensor representing testing samples of benign and spectre assembly code
	"""
	def __init__(self) -> None:
		super().__init__()
		self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.CNN = CNN().to(self.DEVICE)
		self.CRITERION = nn.BCELoss()
		self.OPTIMIZER = optim.Adam(self.CNN.parameters(), lr=0.0002, betas=(0.5, 0.999))
		self.EPOCHS = 10
		self.VALIDATION_LOSS_MIN = np.inf
		self.TRAIN_LOSSES, self.TRAIN_ACCURACIES, self.VALIDATION_LOSSES, self.VALIDATION_ACCURACIES = ([] for _ in range(4))
		self.EPOCH_TRAIN_LOSSES, self.EPOCH_TRAIN_ACCURACIES, self.EPOCH_VALIDATION_LOSSES, self.EPOCH_VALIDATION_ACCURACIES = ([] for _ in range(4))
	
	def normalize_tensor(self, tensor):
		tensor_transform = tensor.reshape(-1, 2)
		tensor_transform -= tensor_transform.mean(axis = 0)
		tensor_transform /= tensor_transform.std(axis = 0)
		return tensor_transform.reshape(tensor.shape) 

	def training_set(self) -> Tuple[Tensor, Tensor]:
		training_embeddings, training_labels, _, _, _, _ = self.data_transfrom()
		training_x = self.normalize_tensor(torch.as_tensor(training_embeddings))
		training_y = torch.as_tensor(training_labels)
		return training_x, training_y

	def validation_set(self) -> Tuple[Tensor, Tensor]:
		_, _, validation_embeddings, validation_labels, _, _ = self.data_transfrom()
		validation_x = self.normalize_tensor(torch.as_tensor(validation_embeddings))
		validation_y = torch.as_tensor(validation_labels)
		return validation_x, validation_y

	def test_set(self) -> Tuple[List, List]:
		_, _, _, _, testing_embeddings, testing_labels = self.data_transfrom()
		test_x = self.normalize_tensor(torch.as_tensor(testing_embeddings))
		test_y = torch.as_tensor(testing_labels)
		return test_x, test_y

	def data_loader(self):
		training_x, training_y = self.training_set()
		validation_x, validation_y = self.validation_set()
		test_x, test_y = self.test_set()
		loader_train = DataLoader(TensorDataset(training_x, training_y), batch_size=64)
		loader_val = DataLoader(TensorDataset(validation_x, validation_y), batch_size=64)
		loader_test = DataLoader(TensorDataset(test_x, test_y), batch_size=64)
		return loader_train, loader_val, loader_test

	def accuracy(self):
		self.CNN.eval()
		correct = 0
		total = 0
		_, _, loader_test = self.data_loader()
		for batch in loader_test:
			observations, labels = batch.to(self.DEVICE)
			outputs = self.CNN(observations)
			_, predicted = torch.max(outputs.data, dim = 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			accuracy = 100 * (correct / total)
		return accuracy

	def train_eval(self):
		loader_train, loader_val, _ = self.data_loader()
		start_time = time.time()
		for epoch in range(self.EPOCHS):
			training_loss = 0.0
			validation_loss = 0.0

			self.CNN.train()
			for idx, batch in tqdm(enumerate(loader_train, 0)):
				observations, labels = batch.to(self.DEVICE)
				self.OPTIMIZER.zero_grad()
				outputs = self.CNN(observations)
				loss = self.CRITERION(outputs, labels)
				loss.backward()
				self.OPTIMIZER.step()
				training_loss += loss.item()

				if idx % 100 == 0:
					print(f"Training Loss = {loss.item()}")
					self.TRAIN_LOSSES.append(loss.item())
					print(f"Training Accuracy = {self.accuracy()}")
					self.TRAIN_ACCURACIES.append(self.accuracy())

			self.EPOCH_TRAIN_LOSSES.append(loss.item())
			self.EPOCH_TRAIN_ACCURACIES.append(self.accuracy())

			self.CNN.eval()
			for idx, batch in tqdm(enumerate(loader_val, 0)):
				observations, labels = batch.to(self.DEVICE)
				outputs = self.CNN(observations)
				loss = self.CRITERION(outputs, labels)
				validation_loss += loss.item()

				if idx % 2.5 == 0:
					print(f"Validation Loss = {loss.item()}")
					self.VALIDATION_LOSSES.append(loss.item())
					print(f"Validation Accuracy = {self.accuracy()}")
					self.VALIDATION_ACCURACIES.append(self.accuracy())

			self.EPOCH_VALIDATION_LOSSES.append(loss.item())
			self.EPOCH_VALIDATION_ACCURACIES.append(self.accuracy())

			print(f"Average training loss: {training_loss} \nAverage validation loss: {validation_loss}")
			print(f"Epoch: {epoch}")
			print(f"Finshed training. \nTime to train: {time.time() - start_time}")
			torch.save(self.CNN.state_dict(), "./model.pth")

	def test(self):
		correct = 0
		total = 0
		_, _, loader_test = self.data_loader()
		with torch.no_grad():
			for batch in loader_test:
				observations, labels = batch.to(self.DEVICE)
				outputs = self.CNN(observations)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
				acc = 100 * (correct/total)
			return(f"Current best test accuracy: {acc}")
	
	def save_metrics(self):
		self.TRAIN_LOSSES, self.TRAIN_ACCURACIES, self.VALIDATION_LOSSES, self.VALIDATION_ACCURACIES 
		metrics = {
			"TRAIN_LOSSES" : self.TRAIN_LOSSES, 
			"TRAIN_ACCURACIES" : self.TRAIN_ACCURACIES,
			"VALIDATION_LOSSES" : self.VALIDATION_LOSSES,
			"VALIDATION_ACCURACIES" : self.VALIDATION_ACCURACIES,
			"TRAIN_LOSSES" : self.TRAIN_LOSSES,
			"TRAIN_ACCURACIES" : self.TRAIN_ACCURACIES,
			"VALIDATION_LOSSES" : self.VALIDATION_LOSSES,
			"VALIDATION_ACCURACIES" : self.VALIDATION_ACCURACIES
		}
		self.pickle(metrics, os.getcwd() + "/CNN/" + "metrics.pickle")

if __name__ == "__main__":
	TrainTest().train_eval()
	TrainTest().test()
	TrainTest().save_metrics()