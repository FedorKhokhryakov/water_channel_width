"""Экспорт весов в ONNX (+ model.json с параметрами препроцессинга) и сверка с torch.

    python training/export_onnx.py --ckpt weights/best.pth --out models/model.onnx [--int8 DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import UNet  # noqa: E402

from icechannel.segmenter import DEFAULT_META, preprocess  # noqa: E402


def load_model(ckpt: str, n_classes: int, base_c: int) -> torch.nn.Module:
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = obj.get("model_state", obj) if isinstance(obj, dict) else obj
    cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
    m = UNet(n_classes=cfg.get("n_classes", n_classes), in_channels=3, base_c=cfg.get("base_c", base_c))
    m.load_state_dict(state)
    return m.eval(), cfg


def _images(folder: str, n: int):
    from PIL import Image

    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    step = max(1, len(files) // n)
    for p in files[::step][:n]:
        yield np.asarray(Image.open(p).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="models/model.onnx")
    ap.add_argument("--base-c", type=int, default=32)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--size", type=int, nargs=2, default=None, metavar=("H", "W"))
    ap.add_argument("--resize", choices=["linear", "area"], default=None,
                    help="как ресайзились кадры при обучении")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--int8", metavar="CALIB_DIR", help="папка кадров для статического int8-квантования")
    ap.add_argument("--check-images", metavar="DIR", help="сверить torch и onnx на реальных кадрах")
    a = ap.parse_args()

    model, cfg = load_model(a.ckpt, a.classes, a.base_c)
    meta = dict(DEFAULT_META)
    meta["input_size"] = list(a.size or cfg.get("size") or DEFAULT_META["input_size"])
    meta["resize"] = a.resize or cfg.get("resize") or "linear"
    h, w = meta["input_size"]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, h, w)
    torch.onnx.export(model, dummy, str(out), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=a.opset, do_constant_folding=True, dynamo=False)

    import onnxruntime as ort

    def compare(path: Path, imgs) -> float:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        agree = []
        for img in imgs:
            x = preprocess(img, meta)
            with torch.no_grad():
                ref = model(torch.from_numpy(x)).numpy()
            got = sess.run(None, {"image": x})[0]
            agree.append(float((ref.argmax(1) == got.argmax(1)).mean()))
            print("  max|Δлогит| = %.2e, совпадение маски %.4f" % (np.abs(ref - got).max(), agree[-1]))
        return float(np.mean(agree))

    rng = np.random.RandomState(0)
    test_imgs = list(_images(a.check_images, 5)) if a.check_images else [
        (rng.rand(1440, 2560, 3) * 255).astype(np.uint8)]
    print("Проверка fp32:")
    compare(out, test_imgs)

    if a.int8:
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter([{"image": preprocess(img, meta)} for img in _images(a.int8, 64)])

            def get_next(self):
                return next(self.it, None)

        q = out.with_name(out.stem + "_int8.onnx")
        quantize_static(str(out), str(q), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        print("Проверка int8 (%s, %.1f МБ):" % (q.name, q.stat().st_size / 1e6))
        agree = compare(q, list(_images(a.check_images or a.int8, 8)))
        print("Среднее совпадение масок int8/fp32: %.4f (используйте int8, если ≥ 0.995)" % agree)
        q.with_suffix(".json").write_text(json.dumps(dict(meta, version=_version(q)), indent=1, ensure_ascii=False))

    meta["version"] = _version(out)
    meta["source_ckpt"] = Path(a.ckpt).name
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print("Готово: %s (%.1f МБ) + %s" % (out, out.stat().st_size / 1e6, out.with_suffix(".json").name))


def _version(p: Path) -> str:
    sha = hashlib.sha256(p.read_bytes()).hexdigest()[:8]
    return "%s-%s" % (datetime.now().strftime("%Y%m%d"), sha)


if __name__ == "__main__":
    main()
