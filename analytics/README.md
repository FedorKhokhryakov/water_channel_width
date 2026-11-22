# analytics

`baseline/` — прежний расчёт ширины (3382 кадра, октябрь–ноябрь 2024): итоговый CSV и label-маски,
по которым он считался. `compare_baseline.py` пересчитывает эти маски текущей геометрией и сравнивает.

`compression_report.py` — сводка баллов сжатия по дням и график (`docs/COMPRESSION.md`).

`nmea_speed.py` — скорость судна из логов icescanner в архивах (для `process --speed-csv`).
