import os
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from typing import Dict, List, Sequence, Set, Text, Tuple, Union

class HandleSpectreData:
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

class DataTransform(HandleSpectreData):
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

class SpectreEmbedding(DataTransform):
	""" Documentation here! """
	def __init__(self) -> None:
		super().__init__()

	def train_temp(self):
		# main ==> self.benign_spectre_train()
		return [['0000000000000000', '<ff_mqc_initdec>:', 'push', '%rbx'], ['mov', '%rdi,%rbx', 'sub', '$0x10,%rsp', 'mov'],
		['test', '%ecx,%ecx', '<ff_mqc_initdec+0x1e>'], ['mov', '%rsi,0x8(%rsp)', '19', '<ff_mqc_initdec+0x19>'], 
		['mov', '0x8(%rsp),%rsi', 'mov', '%rsi,(%rbx)'], ['movzbl', '(%rsi),%eax', 'not', '%al,%eax'], 
		['%edx,0x30(%rdi)', '60', 'lea', '<ff_mqc_initdec+0x60>'], ['shl', '$0x10,%eax', 'mov', '%eax,0x14(%rbx)', 'cmpb'], 
		['$0xff,(%rsi)', 'je', '0x1(%rsi),%rdx'], ['je', '1e', 'callq', '%eax', 'movzbl']]
	
	def train_temp_targets(self): return [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
	
	def model(self) -> None:
		vec = Word2Vec(sentences = self.train_temp(), min_count = 1, vector_size = 32).wv
		vec.save(os.getcwd() + "/CNN/" + "training.wordvectors")
	
	def generator(self) -> List[List[float]]:
		vec_dict = {}
		model = KeyedVectors.load(os.getcwd() + "/CNN/" + "training.wordvectors", mmap = "r")
		for key in model.key_to_index.keys():
			vec_dict[key] = model[key]
		node_vecs = []
		for node_vec in self.train_temp():
			temp = []
			if node_vec is not None:
				for node in node_vec:
					if node in vec_dict:
						temp.append(vec_dict.get(node))
			node_vecs.append(temp)
		return node_vecs

	def flatten(self) -> List[float]:
		flatten_vecs = []

		for vector_list in self.generator():
			if not vector_list:
				flatten_vecs.append(vector_list)
			else:
				flatten_list = np.concatenate(vector_list).ravel().tolist()
				flatten_vecs.append(flatten_list) 

		return flatten_vecs
		
	def upsample_input(self) -> Tuple[List[float], int]:
		out = []
		max_length = max([self.__len__(sub_list) for sub_list in self.flatten()]) 
		for sublist in self.flatten(): 
			if not sublist:
				out.append(np.zeros(max_length).tolist())
			elif len(sublist) < max_length:
				sublist.extend(np.zeros(max_length - self.__len__(sublist)))
				out.append(sublist)
			else:
				out.append(sublist)
		return out, max_length
