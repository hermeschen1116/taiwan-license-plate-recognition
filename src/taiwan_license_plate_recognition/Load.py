from optimum.intel import OVModelForVision2Seq, OVWeightQuantizationConfig
from transformers import TrOCRProcessor
from transformers.pipelines import ImageToTextPipeline
from ultralytics import YOLO


def load_YOLO(model_path: str):
	return YOLO(model_path, task="obb")


def load_TrOCR_model(
	model_path: str,
	quantization_config=OVWeightQuantizationConfig(),
	ov_config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": ".ov_cache"},
	device: str = "cpu",
):
	return OVModelForVision2Seq.from_pretrained(
		model_path, export=True, ov_config=ov_config, quantization_config=quantization_config, device=device
	)


def load_TrOCR_processor(model_path: str):
	return TrOCRProcessor.from_pretrained(model_path, clean_up_tokenization_spaces=True)


def load_TrOCR(model, processor, num_workers: int, device: str):
	return ImageToTextPipeline(
		model=model,
		tokenizer=processor.tokenizer,
		image_processor=processor,
		framework="pt",
		task="image-to-text",
		num_workers=num_workers,
		device=device,
		torch_dtype="auto",
	)
