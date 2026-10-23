# training

Обучение и экспорт модели (нужен torch): `pip install -r training/requirements.txt`.

```bash
python training/train.py --images data/stern/images --masks data/stern/masks --out checkpoints
python training/export_onnx.py --ckpt checkpoints/best.pth --out models/model.onnx
python training/evaluate.py
```

| файл | что делает |
|---|---|
| `model.py` | U-Net |
| `dataset.py` | датасет и разбиение train/val |
| `train.py` | обучение |
| `export_onnx.py` | экспорт в ONNX для судна |
| `evaluate.py` | IoU воды и ошибка ширины против разметки |
