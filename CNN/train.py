import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from config import Config
import torch.optim as optim
from spectre_cnn import SpectreCNN
from spectre_embed import SpectreEmbedding
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(100)

CONFIG = Config()
parser = argparse.ArgumentParser(description = "Trains the SpectreCNN model")
parser.add_argument('--epochs', type = int, default = CONFIG.EPOCHS, help = "Number of training cycles")
parser.add_argument('--lr', type = float, default = CONFIG.LEARNING_RATE, help = "Adam optimizer learning rate")
parser.add_argument('--batch_size', type = int, default = CONFIG.BATCH_SIZE, help = "Number of samples propagated through SpectreCNN")
args = parser.parse_args()

class Train(SpectreEmbedding):
	"""
	Trains the Convolutional Neural Network (CNN) Architecture.
	Args: 
		SpectreCNN: A CNN architecture.
		train: A tensor representing training samples of benign and spectre embeddings.
		validation:A tensor representing validation samples of benign and spectre embeddings.
		test: A tensor representing testing samples of benign and spectre embeddings.
	"""
	def __init__(self) -> None:
		super().__init__()
		self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.SPECTRE_CNN = SpectreCNN().to(self.DEVICE)
		self.CRITERION = nn.BCELoss()
		self.OPTIMIZER = optim.Adam(self.SPECTRE_CNN.parameters(), lr = args.lr, betas = CONFIG.BETAS)
		self.TRAIN_LOSSES, self.TRAIN_ACCURACIES, self.VALIDATION_LOSSES, self.VALIDATION_ACCURACIES = ([] for _ in range(4))
		self.EPOCH_TRAIN_LOSSES, self.EPOCH_TRAIN_ACCURACIES, self.EPOCH_VALIDATION_LOSSES, self.EPOCH_VALIDATION_ACCURACIES = ([] for _ in range(4))
	
	def normalize_tensor(self, tensor):
		tensor_transform = tensor.reshape(-1, 2)
		tensor_transform -= tensor_transform.mean(axis = 0)
		tensor_transform /= tensor_transform.std(axis = 0)
		return tensor_transform.reshape(tensor.shape) 

	def data_loader(self):
		training_x, training_y, validation_x, validation_y, test_x, test_y = self.data_transfrom()
		loader_train = DataLoader(TensorDataset(torch.from_numpy(training_x), torch.from_numpy(training_y)), shuffle = True, batch_size = args.batch_size)
		loader_val = DataLoader(TensorDataset(torch.from_numpy(validation_x), torch.from_numpy(validation_y)), shuffle = True, batch_size = args.batch_size)
		loader_test = DataLoader(TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)), shuffle = True, batch_size = args.batch_size)
		return loader_train, loader_val, loader_test
	
	def accuracy(self):
		accurate_predictions = 0
		_, _, loader_test = self.data_loader()
		self.SPECTRE_CNN.eval()
		for embeddings, labels in loader_test:
			outputs = self.SPECTRE_CNN(embeddings.to(self.DEVICE).unsqueeze(1).float())
			predictions = torch.round(outputs.squeeze()).eq(labels.to(self.DEVICE).float().view_as(torch.round(outputs.squeeze())))
			accurate_predictions += np.sum(np.squeeze(predictions.numpy()))
		return accurate_predictions / self.__len__(loader_test.dataset) * 100

	def train(self):
		loader_train, loader_val, _ = self.data_loader()
		start_time = time.time()
		print(">_ training")
		for epoch in range(args.epochs):
			training_loss = []
			validation_loss = []

			self.SPECTRE_CNN.train()
			for idx, batch in tqdm(enumerate(loader_train, 0)):
				embeddings, labels = batch
				self.OPTIMIZER.zero_grad()

				outputs = self.SPECTRE_CNN(embeddings.to(self.DEVICE).unsqueeze(1).float())
				loss = self.CRITERION(outputs.squeeze(), labels.to(self.DEVICE).float())
				loss.backward()
				self.OPTIMIZER.step()
				training_loss.append(loss.item())
				if idx % 50 == 0: 
					print(f"Training Loss = {loss.item()}")
					self.TRAIN_LOSSES.append(loss.item())
					print(f"Training Accuracy = {self.accuracy()}")
					self.TRAIN_ACCURACIES.append(self.accuracy())

			self.EPOCH_TRAIN_LOSSES.append(loss.item())
			self.EPOCH_TRAIN_ACCURACIES.append(self.accuracy())

			self.SPECTRE_CNN.eval()
			for idx, batch in tqdm(enumerate(loader_val, 0)):
				embeddings, labels = batch
				outputs = self.SPECTRE_CNN(embeddings.to(self.DEVICE).unsqueeze(1).float())
				loss = self.CRITERION(outputs.squeeze(), labels.to(self.DEVICE).float())
				validation_loss.append(loss.item())
				if idx % 10 == 0:
					print(f"Validation Loss = {loss.item()}")
					self.VALIDATION_LOSSES.append(loss.item())
					print(f"Validation Accuracy = {self.accuracy()}")
					self.VALIDATION_ACCURACIES.append(self.accuracy())

			self.EPOCH_VALIDATION_LOSSES.append(loss.item())
			self.EPOCH_VALIDATION_ACCURACIES.append(self.accuracy())

			print(f"Average training loss: {np.mean(training_loss)} \nAverage validation loss: {np.mean(validation_loss)}")
			print(f"Epoch: {epoch}")
			print(f"Finshed training. \nTime to train: {time.time() - start_time} seconds")
		torch.save(self.SPECTRE_CNN.state_dict(), "./model.pth")
	
	def store_metrics(self):
		metrics = {
			"train_losses" : self.TRAIN_LOSSES, 
			"train_accuracies" : self.TRAIN_ACCURACIES,
			"validation_losses" : self.VALIDATION_LOSSES,
			"validation_accuracies" : self.VALIDATION_ACCURACIES,
			"epoch_train_losses" : self.EPOCH_TRAIN_LOSSES,
			"epoch_train_accuracies" : self.EPOCH_TRAIN_ACCURACIES,
			"epoch_validation_losses" : self.EPOCH_VALIDATION_LOSSES,
			"epoch_validation_accuracies" : self.EPOCH_VALIDATION_ACCURACIES
		}
		self.pickle(metrics, "./metrics.pickle")