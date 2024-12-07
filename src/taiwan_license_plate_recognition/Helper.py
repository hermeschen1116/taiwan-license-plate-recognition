import multiprocessing

import torch


def get_torch_device() -> str:
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"

	return "cpu"


def get_num_of_workers() -> int:
	return multiprocessing.cpu_count()


def accuracy_metric(predictions, references) -> float:
	corresponds = [int(p == t) for (p, t) in zip(predictions, references)]
	return sum(corresponds) / len(corresponds)
