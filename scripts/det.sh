uv run src/scripts/recognition/PaddleOCR/tools/train.py -c src/scripts/recognition/arguments/ch_PP-OCRv3_det_student.yml
uv run src/scripts/recognition/PaddleOCR/tools/export_model.py -c src/scripts/recognition/arguments/ch_PP-OCRv3_det_student.yml -o Global.pretrained_model="./output/en_PP-OCRv3_det_slim_distill/best_accuracy" Global.use_wandb=False
uv run src/scripts/recognition/evaluate_PaddleOCR.py
