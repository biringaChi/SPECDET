import torch
import numpy as np
from train import Train
import matplotlib.pyplot as plt
from spectre_embed import SpectreEmbedding
from sklearn.metrics import precision_score, recall_score, f1_score
plt.style.use("ggplot")

alpha = 0.2
markersize = 5
figure, ax = plt.subplots()
fontsize = 20

class Test(Train):
	"""
	Test the Convolutional Neural Network (CNN) Architecture.
	Args: 
	SpectreCNN Model.
	"""
	def __init__(self) -> None:
		super().__init__()

	def load_metrics(self):
		return SpectreEmbedding().unpickle("CNN/metrics.pickle")
		
	def get_training_metrics(self):
		metrics = self.load_metrics()
		train_losses = metrics["train_losses"][:20]
		validation_losses = metrics["validation_losses"][:20]
		index = [i for i in range(len(train_losses))]
		return index, train_losses, validation_losses

	def test_accuracy(self):
		test_losses = [] 
		accurate_predictions = 0
		_, _, loader_test = self.data_loader()
		with torch.no_grad():
			for embeddings, labels in loader_test:
				outputs = self.SPECTRE_CNN(embeddings.to(self.DEVICE).unsqueeze(1).float())
				test_loss = self.CRITERION(outputs.squeeze(), labels.to(self.DEVICE).float())
				test_losses.append(test_loss.item())
				predictions = torch.round(outputs.squeeze()).eq(labels.to(self.DEVICE).float().view_as(torch.round(outputs.squeeze())))
				accurate_predictions += np.sum(np.squeeze(predictions.numpy()))
			avg_test_loss = np.mean(test_losses)
			return(f"Average Test Accuracy: {accurate_predictions / self.__len__(loader_test.dataset)}")
	
	def evaluate(self):
		self.SPECTRE_CNN.eval()
		_, _, loader_test = self.data_loader()
		labels_metric, prediction_metric = ([] for _ in range(2))

		with torch.no_grad():
			for embeddings, labels in loader_test:
				outputs = self.SPECTRE_CNN(embeddings.unsqueeze(1).float())
				predictions = torch.round(outputs.squeeze())
				prediction_metric.append(predictions.tolist())
				labels_metric.append(labels.tolist())

		labels_metric  = [i for sublist in labels_metric for i in sublist]
		prediction_metric  = [i for sublist in prediction_metric for i in sublist]
		precision = precision_score(labels_metric, prediction_metric)
		recall = recall_score(labels_metric, prediction_metric)
		f1 = f1_score(labels_metric, prediction_metric)
		return precision, recall, f1

class Visualization(Test):
	def __init__(self) -> None:
		super().__init__()

	def train_validation_loss(self):
		index, train_losses, validation_losses = self.get_training_metrics()
		ax.plot(index, train_losses, marker = "*", markersize = markersize, label = "train", color = "#2471A3")
		ax.plot(index, validation_losses, marker = "o", markersize = markersize, label = "validation", color = "#196F3D")
		ax.set_ylabel("Loss", fontsize = fontsize)
		ax.set_xlabel("Iteration", fontsize = fontsize)

		ax.legend(fontsize = fontsize)
		plt.xticks(fontsize = fontsize, rotation = 45)
		plt.yticks(fontsize = fontsize)
		plt.tight_layout() 
		plt.show()