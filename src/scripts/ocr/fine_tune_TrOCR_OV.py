import os

from PIL import Image
from dotenv import load_dotenv
from optimum.intel import OVModelForVision2Seq
from transformers import TrOCRProcessor

from taiwan_license_plate_recognition.helper import get_num_of_workers, get_torch_device

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()
device: str = get_torch_device()
max_length: int = 64

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", clean_up_tokenization_spaces=True)

test_image_path: str = f"{project_root}/datasets/ocr/高雄市ZS-0786.jpg"

image = Image.open(test_image_path)

model = OVModelForVision2Seq.from_pretrained(
	"DunnBC22/trocr-base-printed_license_plates_ocr", export=True, device="cpu"
)

encode_image = processor(image.convert("RGB"), return_tensors="pt").pixel_values

generated_ids = model.generate(encode_image, max_length=max_length)

result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(result)
