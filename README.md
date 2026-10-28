# water_channel_width

Ширина ледового канала за кормой ледокола по кадрам кормовой камеры.

```
кадр → U-Net (ONNX) → горизонт: наклон и крен камеры → кромки канала → ширина в метрах
```

Сервис будет работать на судне (Astra Linux, без интернета): следит за папкой, куда судовая система
складывает zip-архивы `ДАТА_ВРЕМЯ_stern_lidar.zip`, и пишет результаты в свою папку.

## Камера

Кормовая камера, кадр 2560×1440, высота над водой 21 м (`docs/icebreaker_geometry.jpg`),
калибровка в `calibration/stern_camera.yaml`.

## Структура

```
calibration/      калибровка камеры: снимки доски, скрипт, stern_camera.yaml
data/stern/       ручная разметка: 145 кадров и масок (формат CVAT)
training/         обучение U-Net, экспорт в ONNX, оценка (нужен torch)
src/icechannel/   пакет для судна: numpy, onnxruntime, Pillow, PyYAML
config/           пример конфига сервиса
analytics/        сравнение с прежним расчётом ширины
packaging/        офлайн-установка на Astra Linux, служба systemd
```

## Разработка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e . -r training/requirements.txt
python training/export_onnx.py --ckpt weights/best.pth --out models/model.onnx   # веса — в Releases
python training/evaluate.py                                                     # проверка на разметке
python -m icechannel process -c config/config.example.yaml -o out/test data/stern/images
```

Обучение с нуля: `python training/train.py --images data/stern/images --masks data/stern/masks --out checkpoints`.

## Установка на судно

```bash
packaging/build_bundle.sh 3.7            # на машине с интернетом; 3.7 — версия Python на судне
# скопировать bundle/ice-channel-*.tar.gz на судно, затем:
tar xzf ice-channel-*.tar.gz && cd ice-channel-*/
sudo ./install.sh --systemd              # в /opt/ice-channel
sudo nano /opt/ice-channel/config/config.yaml       # paths.inbox, paths.output
sudo systemctl start ice-channel
```

На судне: numpy, onnxruntime, Pillow, PyYAML и модель — около 105 МБ (Python 3.7), без torch и OpenCV.

Веб-страница: `/opt/ice-channel/bin/ice-channel serve -c /opt/ice-channel/config/config.yaml`,
с берега через `ssh -L 8585:127.0.0.1:8585 user@ship` и http://localhost:8585.

## Результаты

`output/results/ГГГГ-ММ-ДД.csv`, одна строка на кадр:

| поле | что это |
|---|---|
| `width_m` | ширина канала до 250 м за кормой, перпендикулярно оси канала |
| `width_smooth_m` | медиана последних надёжных кадров |
| `w_100m … w_300m` | ширина в сечениях |
| `status`, `flags` | `ok` / `suspect` / `fail` и причины |
| `pitch_deg`, `roll_deg` | наклон и крен камеры по горизонту |
| `vessel_z_m` | расстояние до судна, идущего за ледоколом по каналу |

Рядом лежат контрольные картинки (`overlays/`), маски и лог.
