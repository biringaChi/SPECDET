import os
import pickle
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from typing import Dict, List, Sequence, Set, Text, Tuple, Union

class HandleSpectreBenignData:
	"""
	DataHandler: Handles spectre and benign data.
	Args: 
		benign_train: A string representing training samples of benign assembly code.
		spectre_train: A string representating training samples of spectre gadget. 
		benign_test: A string representing testing samples of benign assembly code.
		spectre_test: A string representating testing samples of spectre gadget. 
	"""
	def __init__(
		self,
		BENIGN_TRAIN_PATH: str = os.getcwd() + "/datasets/spectre_gadgets/benign_train",  
		SPECTRE_TRAIN_PATH: str = os.getcwd() + "/datasets/spectre_gadgets/spectre_train",
		BENIGN_TEST_PATH: str = os.getcwd() + "/datasets/spectre_gadgets/benign_test",
		SPECTRE_TEST_PATH: str = os.getcwd() + "/datasets/spectre_gadgets/spectre_test"
		) -> None:
		super().__init__()
		self.BENIGN_TRAIN_PATH = BENIGN_TRAIN_PATH
		self.SPECTRE_TRAIN_PATH = SPECTRE_TRAIN_PATH
		self.BENIGN_TEST_PATH = BENIGN_TEST_PATH
		self.SPECTRE_TEST_PATH = SPECTRE_TEST_PATH
		self.DATA_REPLACE = {"DR1" : "Disassembly"}
		
	def __len__(self, arg: Union[Sequence, Text, Dict, Set]) -> int:
		if (isinstance(arg, (int, float, bool))):
			raise TypeError("Invalid argument. Only text, sequence, mapping and set are accepted")
		else:
			return len(arg)

	def get_assembly(self, path_arg) -> List[str]:
		assembly = []
		for root, _, files in os.walk(path_arg):
			for file in files:
				if file.endswith(".s"):
					temp = os.path.join(root, file)
					try:
						with open(temp, "r") as assembly_code:
							assembly.append(assembly_code.read())
					except OSError as e:
						raise e
		return assembly
	
	def benign_train(self) -> List[str]:
		return self.get_assembly(self.BENIGN_TRAIN_PATH)
	
	def benign_train_targets(self) -> List[str]:
		return ["benign" for _ in range(self.__len__(self.benign_train()))]

	def spectre_train(self):
		return self.get_assembly(self.SPECTRE_TRAIN_PATH)

	def spectre_train_targets(self) -> List[str]:
		return ["spectre" for _ in range(self.__len__(self.spectre_train()))]
	
	def benign_test(self) -> List[str]:
		return self.get_assembly(self.BENIGN_TEST_PATH)
	
	def benign_test_targets(self) -> List[str]:
		return ["benign" for _ in range(self.__len__(self.benign_test()))]
	
	def spectre_test(self) -> List[str]:
		return self.get_assembly(self.SPECTRE_TEST_PATH)
	
	def spectre_test_targets(self) -> List[str]:
		return ["spectre" for _ in range(self.__len__(self.spectre_test()))]

class DataTransform(HandleSpectreBenignData):
	"""
	Data Transform: transforms handled data into DNN algorithmic shape.
    Args: 
		spectre: A list of strings representing raw spectre vulnerable code snippets.
		benign: A list of strings representing raw assembly code snippets.
	"""

	def __init__(self) -> None:
		super().__init__()

	def wrangle(self, data) -> List[str]:
		out = []
		for benign_spectre in data:
			temp = []
			for line in benign_spectre.split("\n")[2:]:
				if line.startswith(self.DATA_REPLACE["DR1"]):
					temp.append(line.replace(line, ""))
				else: temp.append(line)
			out.append(" ".join(temp))
		return out
	
	def encoder(self, data_sample, sample_val, data = None) -> List[int]:
		out = []
		if data is not None:
			for target in data:
				if target == "benign":
					out.append(0)
				else: out.append(1)
		else: 
			for idx, _ in enumerate(data_sample):
				if idx < sample_val:
					out.append(0)
				else: out.append(1) 
		return out

	def benign_spectre_train(self) -> List[List[str]]:
		benign_spectre_train = self.benign_train() + self.spectre_train()
		return  [data.split() for data in self.wrangle(benign_spectre_train)]

	def benign_spectre_train_targets(self) -> List[int]:
		benign_spectre_train_targets = self.benign_train_targets() + self.spectre_train_targets()
		return self.encoder(benign_spectre_train_targets)
	
	def benign_spectre_test(self) -> List[str]:
		benign_spectre_test = self.benign_test() + self.spectre_test()
		return [data.split() for data in self.wrangle(benign_spectre_test)]

	def benign_spectre_test_targets(self) -> List[int]: 
		benign_spectre_test_targets = self.benign_test_targets() + self.spectre_test_targets()
		return self.encoder(benign_spectre_test_targets)

class Embedding(DataTransform):
	"""
	Neural Embeddings: generates neural embeddings for SpectreCNN.
	Arg: transformed spectre and benign data.
	"""
	
	def __init__(self) -> None:
		super().__init__()
	
	def model(self, data, vec_name) -> None:
		vec = Word2Vec(sentences = data, min_count = 1, vector_size = 32).wv
		vec.save(os.getcwd() + "/CNN/" + vec_name)

	def embed_lookup(self, vec, data) -> np.ndarray:
		vec_dict = {}
		model = KeyedVectors.load(os.getcwd() + "/CNN/" + vec, mmap = "r")
		for key in model.key_to_index.keys():
			vec_dict[key] = model[key]
		out = []
		for embeds in data:
			temp = []
			for embed in embeds:
				if embed in vec_dict:
					temp.append(vec_dict[embed])
			out.append(temp)
		return out

	def flatten(self, data) -> List[float]:
		out = []
		for vector_list in data:
			if not vector_list:
				out.append(vector_list)
			else:
				flatten_list = np.concatenate(vector_list).ravel().tolist()
				out.append(flatten_list) 
		return out
		
	def upsample(self, data) -> Tuple[List[float]]:
		out = []
		max_length = max([self.__len__(sub_list) for sub_list in data]) 
		for sublist in data: 
			if not sublist:
				out.append(np.zeros(max_length).tolist())
			elif self.__len__(sublist) < max_length:
				sublist.extend(np.zeros(max_length - self.__len__(sublist)))
				out.append(sublist)
			else:
				out.append(sublist)
		return out
	
	def training(self) -> np.ndarray:
		return self.upsample(self.flatten(self.embed_lookup("training.wordvectors", self.benign_spectre_train())))

	def testing(self) -> np.ndarray:
		return self.upsample(self.flatten(self.embed_lookup("testing.wordvectors", self.benign_spectre_test())))

class SpectreEmbedding(Embedding):
	"""
	SpectreEmbedding: samples 60,000 observations.
	49,000 Training.
	1,000 Validation.
	10,000 Testing.
	"""
	def __init__(self) -> None:
		super().__init__()
		self.BENIGN_NUM: int = 1481
		self.BENIGN_NUM_VAL: int = 100
		self.BENIGN_NUM_TEST: int = 396
		self.SPECTRE_NUM: int = 47519
		self.SPECTRE_NUM_TEST: int = 9604
		self.SAMPLE: int = 1000
	
	def get_targets(self, data, split_val) -> List[int]:
		targets = []
		for idx, _ in enumerate(data):
			if idx <= split_val:
				targets.append("spectre")
			else: targets.append("benign")
		return self.encoder(targets)
	
	def pickle(self, data, file_name):
		with open(file_name, 'wb') as file:
			pickle.dump(data, file)

	def unpickle(self, data):
		with open(data, "rb") as file:
			loaded = pickle.load(file)
		return loaded
	
	def pad(self, embeddings) -> np.ndarray:
		zeros: np.ndarray = np.zeros((32,), dtype=np.float64)
		out  = []
		max_length: int = max(self.__len__(embedding) for embedding in embeddings)
		for embedding in embeddings:
			if self.__len__(embedding) < max_length:
				embedding.extend([0.0] * (max_length - self.__len__(embedding)))
		for embedding in embeddings:
			temp = []
			for vector in embedding:
				if not type(vector).__module__ == np.__name__:
					temp.append(zeros)
				else: temp.append(vector)
			out.append(temp)
		return out
	
	def train_val_test_set(self) -> np.ndarray:
		out_train, out_val, out_test = ([] for _ in range(3))
		training_set: List[str] = [data.split() for data in self.wrangle(self.benign_train()[:self.BENIGN_NUM] + self.spectre_train()[:self.SPECTRE_NUM])]
		validation_set: List[str] = [data.split() for data in self.wrangle(self.benign_train()[:self.BENIGN_NUM] + self.spectre_train()[:self.SPECTRE_NUM])]
		test_set: List[str] = [data.split() for data in self.wrangle(self.benign_test() + self.spectre_test()[:self.SPECTRE_NUM_TEST])]

		for data_train, data_val, data_test in zip(training_set, validation_set, test_set):
			if self.__len__(data_train) > self.SAMPLE:
				out_train.append(data_train[:self.SAMPLE])
			elif self.__len__(data_train) <= self.SAMPLE:
				out_train.append(data_train)
			if self.__len__(data_val) > self.SAMPLE:
				out_val.append(data_val[:self.SAMPLE])
			elif self.__len__(data_val) <= self.SAMPLE:
				out_val.append(data_val)
			if self.__len__(data_test) > self.SAMPLE:
				out_test.append(data_test[:self.SAMPLE])
			elif self.__len__(data_test) <= self.SAMPLE:
				out_val.append(data_test)
		
		self.model(out_train, "training_set_vectors.wordvectors")
		self.model(out_val, "validation_set_vectors.wordvectors")
		self.model(out_test, "test_set_vectors.wordvectors")
		
		self.pickle(out_train, "training_set.pickle")
		self.pickle(out_val, "validation_set.pickle")
		self.pickle(out_test, "test_set.pickle")
		
		self.pickle(self.encoder(out_train, self.BENIGN_NUM),"training_set_labels.pickle")
		self.pickle(self.encoder(out_val, self.BENIGN_NUM_VAL),"validation_set_labels.pickle")
		self.pickle(self.encoder(out_test, self.BENIGN_NUM_TEST),"test_set_labels.pickle")
		

		training_embedding: np.ndarray = self.embed_lookup(self.unpickle("training_set_vectors.wordvectors"), out_train)
		validation_embedding: np.ndarray = self.embed_lookup(self.unpickle("validation_set_vectors.wordvectors"), out_val)
		test_embedding: np.ndarray = self.embed_lookup(self.unpickle("test_set_vectors.wordvectors"), out_test)

		self.pickle(training_embedding,"training_embedding.pickle")
		self.pickle(validation_embedding,"validation_embedding.pickle")
		self.pickle(test_embedding,"test_embedding.pickle")

	def data_transfrom(self) -> np.ndarray:
		training_embeddings: np.ndarray[float] =  np.asarray(self.pad(self.unpickle(os.getcwd() + "/CNN/data/" + "training_embedding.pickle")))
		training_labels: np.ndarray[int] =  np.asarray(self.unpickle(os.getcwd() + "/CNN/data/" + "training_set_labels.pickle"))

		validation_embeddings: np.ndarray[float] =  np.asarray(self.pad(self.unpickle(os.getcwd() + "/CNN/data/" + "validation_embedding.pickle")))
		validation_labels: np.ndarray[int] = np.asarray(self.unpickle(os.getcwd() + "/CNN/data/" + "validation_set_labels.pickle"))
		
		testing_embeddings: np.ndarray[float] =  np.asarray(self.pad(self.unpickle(os.getcwd() + "/CNN/data/" + "test_embedding.pickle")))
		testing_labels: np.ndarray[int] =  np.asarray(self.unpickle(os.getcwd() + "/CNN/data/" + "test_set_labels.pickle"))

		return training_embeddings, training_labels, validation_embeddings, validation_labels, testing_embeddings, testing_labels