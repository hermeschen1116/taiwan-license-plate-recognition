from typing import List


def accuracy(predictions: List[str], references: List[str]) -> float:
	corresponds = [int(p == t) for (p, t) in zip(predictions, references)]
	return sum(corresponds) / len(corresponds)
