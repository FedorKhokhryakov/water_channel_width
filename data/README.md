# Данные: ручная разметка кормовой камеры

```
data/stern/
  images/        кадры *_stern_cameraupscale.jpg (2560x1440), для которых есть разметка
  masks/         RGB-маски (PNG) с тем же именем кадра (иногда с суффиксом _mask)
  labelmap.txt   классы и цвета в формате CVAT
```

| класс | цвет | что размечено |
|---|---|---|
| background | 0,0,0 | небо и собственная палуба |
| ice | 236,236,242 | лёд по сторонам канала |
| water | 43,169,236 | открытая вода канала |
| ship | 241,47,72 | другие суда (идущие за ледоколом по каналу) |

145 кадров с 18 по 28 октября 2024 г. Разбиение train/val в `training/train.py`: по времени (`--split time`,
последние 20 % кадров) или по дням (`--split day`).

```bash
python training/train.py --images data/stern/images --masks data/stern/masks --out checkpoints_v3
python training/evaluate.py      # IoU воды и ошибка ширины против этой разметки
```
