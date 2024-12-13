# Taiwan License Plate Recognition

## Introduction

## Environment

- Python 3.12
- optimum[nncf,openvino]
- requests

### Detection (YOLO)

- ultralytics==8.0.238
- roboflow (dataset)

### Recogntion (OCR)

- paddlepaddle
- paddleocr

## Architecture

![architecture](architecture.drawio.png)

- First, I use predict API provided by ultralytics and get any supported sources to crop
  license plate images from them.

- Then pass the cropped images to the OCR model to extract license number from the images.

## Usage

- Install dependencies specify in `pyproject.toml`

- Specify environment variables in the `.env`

  ```
  IMAGE_SIZE=640
  YOLO_MODEL_PATH={path to openvino model}
  CAMERA_ADDRESS={path to media source}
  API={api endpoint to send the result}
  ```

- execute main.py under `src/`, then it will start the task.

## Implementation

### Detection

- YOLO series models provide high accuracy and fast object detection, so I choose YOLO for detection task.

- To achive fast and lightweight inference on CPU, I choose a bit old and small model, YOLOv8n.

  > **Why not choose newer model?**
  >
  > When doing research, I found that there's no significant difference on any aspects between v8 and newer model.
  > For detail can check [ultralytics](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes).

- I train the YOLOv8n on OBB task using dataset from roboflow with some modification.

  > **Why OBB task?**
  >
  > OBB format provide rotation of bounding box as extra information, so I can get more precise result.

#### Training

- I use API provided by ultralytics to do hyperparameter-tuning, traing, evaluation.
  There's not so much technique to do these tasks because ultralytics automatically handles theses for us.
  For detail, just check scripts under `src/yolo/`.

- After training, I can easily use export API to save model as OpenVINO IR format for further use.

- Result

<iframe src="https://wandb.ai/hermeschen1116/taiwan-license-plate-recognition/reports/YOLO--VmlldzoxMDU4NDk0MA" style="border:none;height:1024px;width:100%">

### Recognition

## Optimization

- Exporting model to Openvino IR format for fast and efficient inference on CPU so that it can run on weak device.
