# 台灣車牌辨識系統

## 環境

- Python 3.12
- optimum[nncf,openvino]
- requests

### 車牌偵測 (YOLO)

- ultralytics==8.0.238
- roboflow (dataset)

### 車牌辨識 (OCR)

- paddlepaddle
- paddleocr

## 架構

![architecture](architecture.drawio.png)

- 使用 ultralytics 提供的預測 API 來偵測任何支援的影像來源中的車牌，並根據 bounding box 的座標將其從原影像中切出。

- 將裁切出的車牌影像經過 OCR 模型的辨識得到實際的車牌號碼。

## Usage

- 安裝 dependancies，可以直接使用 uv 來安裝。

- 在根目錄下的 `.env` 檔案加入以下的環境變數，根據實際的運行環境跟模型餐數等調整

  ```
  INFERENCE_DEVICE="cpu"
  NUM_WORKERS=8
  FRAME_SIZE=640
  STREAM_SOURCE={ address to stream source }
  DETECTION_MODEL_PATH={ path to model files }
  API_ENDPOINT={ address of api }
  ```

- 執行 `src/` 下的 `main.py` 便會自動運行並回傳結果

## 實作

### 車牌偵測

- 由於 YOLO 系列模型在物體偵測上已經有非常成熟的應用跟表現，故直接採用 YOLO 來作為車牌辨識模型。

- 由於運行環境會是沒有獨立 GPU 的環境，因此為了在 CPU 也能

  > **Why not choose newer model?**
  >
  > When doing research, I found that there's no significant difference on any aspects between v8 and newer model.
  > For detail can check [ultralytics](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes).

- I train the YOLOv8n on OBB task using dataset from roboflow with some modification.

  > **Why OBB task?**
  >
  > OBB format provide rotation of bounding box as extra information, so I can get more precise result.

#### 訓練

- I use API provided by ultralytics to do hyperparameter-tuning, traing, evaluation.
  There's not so much technique to do these tasks because ultralytics automatically handles theses for us.
  For detail, just check scripts under `src/yolo/`.

- After training, I can easily use export API to save model as OpenVINO IR format for further use.

### 車牌辨識

- [比較結果](https://api.wandb.ai/links/hermeschen1116/l16nx6qc)

- First, I try TROCR, a transformer based model for OCR task.
  But it hard to fine tune it on my own dataset.

- So I compare some common OCR methods, EasyOCR, Tesseract, and PaddleOCR.
  And I found PaddleOCR performs the best on my data (a bit obscure).

- And because in Taiwan, we usually can find some stickers and some marks on license plate may contain not related text on them.
  I also implement a `validate_license_number` function to filter out the text in correct format (common 2-4, 4-2, 3-3, and 3-4 format).

#### 訓練

（由於訓練 PaddleOCR 需要進行額外的資料標記，此部分還在進行中）

## Optimization

- Exporting model to Openvino IR format for fast and efficient inference on CPU so that it can run on weak device.
