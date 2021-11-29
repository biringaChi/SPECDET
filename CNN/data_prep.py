import os
import csv
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from typing import Dict, List, Sequence, Set, Text, Tuple, Union

class HandleSpectreBenignData:
	"""
	Handles spectre and benign data
	Args: 
		benign_train: A string representing training samples of benign assembly code
		spectre_train: A string representating training samples of spectre gadget 
		benign_test: A string representing testing samples of benign assembly code
		spectre_test: A string representating testing samples of spectre gadget 
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
		for (root, _, files) in os.walk(path_arg):
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

	def spectre_train_targets(self):
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
	"""Transforms handled data into ML algorithm shape"""

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
	
	def encoder(self, data) -> List[int]:
		out = []
		for target in data:
			if target == "benign":
				out.append(0)
			else: out.append(1)
		return out

	def benign_spectre_train(self) -> List[List[str]]:
		benign_spectre_train = self.benign_train() + self.spectre_train()
		return [data.split() for data in self.wrangle(benign_spectre_train)]

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
	"""Generates embeddings"""

	def __init__(self) -> None:
		super().__init__()
	
	def model(self, data, vec_name) -> None:
		vec = Word2Vec(sentences = data, min_count = 1, vector_size = 32).wv
		vec.save(os.getcwd() + "/CNN/" + vec_name)
	
	def generator(self, vec, data) -> List[List[float]]:
		vec_dict = {}
		model = KeyedVectors.load(os.getcwd() + "/CNN/" + vec, mmap = "r")
		for key in model.key_to_index.keys():
			vec_dict[key] = model[key]
		node_vecs = []
		for node_vec in data:
			temp = []
			if node_vec is not None:
				for node in node_vec:
					if node in vec_dict:
						temp.append(vec_dict.get(node))
			node_vecs.append(temp)
		return node_vecs

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
			elif len(sublist) < max_length:
				sublist.extend(np.zeros(max_length - self.__len__(sublist)))
				out.append(sublist)
			else:
				out.append(sublist)
		return out
	
	def training(self):
		"""
		Main training method to generate 957,673(training) embedding vectors
		-- Problem --> Significantly large file, floods memory.
		"""
		return self.upsample(self.flatten(self.generator("training.wordvectors", self.benign_spectre_train())))

	def testing(self):
		"""
		Main testing method to generate 957,673(testing) embedding vectors
		-- Problem --> Same with training method above
		"""
		return self.upsample(self.flatten(self.generator("testing.wordvectors", self.benign_spectre_test())))

class DataSample60K(Embedding):
	"""Samples 60,000 Observations - (50K training and 10K testing)"""
	def __init__(self) -> None:
		super().__init__()
		self.SPECTRE_TRAIN_SAMPLE = 48_419
		self.SPECTRE_TEST_SAMPLE = 9_604
	
	def sample_train(self):
		spectre_train_sample = self.spectre_train()[:self.SPECTRE_TRAIN_SAMPLE]
		benign_spectre_train_50K = spectre_train_sample + self.benign_train()
		return [data.split() for data in self.wrangle(benign_spectre_train_50K)]
	
	def store_data(self):
		_file_path: str = os.getcwd() + "/CNN/data/benign_spectre_train_50K.csv"
		os.makedirs(os.path.dirname(_file_path), exist_ok = True)
		try:
			with open(_file_path, "w") as file:
				with file:
					write = csv.writer(file)
					write.writerows(self.sample_train())
		except OSError as e:
			raise e
	
	def model(self):
		return self.model(self.sample_train(), "benign_spectre_train_50K.wordvectors")

	def training_sample(self) -> List[List[float]]:
		out = self.upsample(self.flatten(self.generator("benign_spectre_train_50K.wordvectors", self.sample_train())))
		return out
	
	def process_dataset(self) -> None:
		_file_name = "processed_bst_50K.csv"
		try:
			with open(_file_name, "w") as file:
				with file:
					write = csv.writer(file)
					write.writerows(self.training_sample())
		except OSError as e:
			raise e
		
	def get_targets_train(self) -> List[int]:
		targets = []
		for idx, _ in enumerate(self.sample_train()):
			if idx <= self.SPECTRE_TRAIN_SAMPLE:
				targets.append("spectre")
			else: targets.append("benign")
		return self.encoder(targets)


# if __name__ == "__main__":
# 	DataSample60K().process_dataset()