# Taiwan License Plate Recognition

## Environment

- Python 3.12
- optimum[nncf,openvino]

### Detection (YOLO)

- ultralytics==8.0.238
- roboflow

### Recogntion (OCR)

- paddlepaddle
- paddleocr

## Architecture

![architecture](architecture.drawio.png)

- First, I use predict API provided by ultralytics and get any supported sources to crop
  license plate images from them.

- Then pass the cropped images to the OCR model to extract license number from the images.

## Usage

- Install dependencies specify in pyproject.toml

- Specify environment variables in the .env

  ```
  IMAGE_SIZE=640
  YOLO_MODEL_PATH={path to openvino model}
  CAMERA_ADDRESS={path to media source}
  API={api endpoint to send the result}
  ```

- execute main.py under src/, then it will start the task.

## Implementation
