from typing import Dict, Sequence, Set, Text, Union

class Util:
	def __init__(self) -> None:
		pass

	def __str__(self) -> str:
		return f"{self.__class__.__name__}(_)"

	def __repr__(self) -> str:
		return self.__str__()
	
	def __len__(self, arg: Union[Sequence, Text, Dict, Set]) -> int:
		if (isinstance(arg, (int, float, bool))):
			raise TypeError("Invalid argument. Only text, sequence, mapping and set are accepted")
		else:
			return len(arg)