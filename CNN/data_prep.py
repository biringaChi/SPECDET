import os
from typing import List, Union
from typing import Dict, Sequence, Set, Text, Union

class HandleSpectreData:
	def __init__(
		self,
		BENIGN_TRAIN_PATH: str = os.getcwd() + "/datasets/spectre_gadgets/benign_train",  
		SPECTRE_TRAIN_PATH: str = os.getcwd() + "/datasets/spectre_gadgets/spectre_train",
		BENIGN_TEST_PATH: str = os.getcwd() + "/datasets/spectre_gadgets/benign_test",
		SPECTRE_TEST_PATH: str = os.getcwd() + "/datasets/spectre_gadgets/spectre_test",
		) -> None:
		super().__init__()
		self.BENIGN_TRAIN_PATH = BENIGN_TRAIN_PATH
		self.SPECTRE_TRAIN_PATH = SPECTRE_TRAIN_PATH
		self.BENIGN_TEST_PATH = BENIGN_TEST_PATH
		self.SPECTRE_TEST_PATH = SPECTRE_TEST_PATH
		
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

class Data(HandleSpectreData):
	def __init__(self) -> None:
		super().__init__()

	def benign_spectre_train(self) -> List[str]:
		temp = self.benign_train() + self.spectre_train()
		# more wrangling
		pass
			
	def benign_spectre_train_targets(self) -> List[int]:
		temp = self.benign_train_targets() + self.spectre_train_targets()
		# encode targets
		pass
	
	def benign_spectre_test(self) -> List[int]:
		temp = self.benign_test() + self.spectre_test()
		# more wrangling
		pass

	def benign_spectre_test_targets(self) -> List[int]:
		temp = self.benign_test_targets() + self.spectre_test_targets()
		# more wrangling
		pass

	def training(self) -> List[str]:
		return self.benign_spectre_train()
	
	def training_targets(self) -> List[int]:
		return self.benign_spectre_train_targets()

	def testing(self) -> List[str]:
		return self.benign_spectre_test()
	
	def testing(self) -> List[int]:
		return self.benign_spectre_test_targets()


class SpectreEmbedding(Data):
	def __init__(self) -> None:
		super().__init__()
		pass
	


if __name__ == "__main__":
	pass
