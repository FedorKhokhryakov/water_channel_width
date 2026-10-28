#!/usr/bin/env bash
# Офлайн-бандл для судна (собирается на машине с интернетом).
#   packaging/build_bundle.sh 3.7     # версия Python на судне
# -> bundle/ice-channel-<версия>-py<XY>.tar.gz
set -euo pipefail

PYVER="${1:-3.7}"
PLATFORM="${PLATFORM:-x86_64}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$(python3 -c "import re;print(re.search(r'__version__ = \"(.+?)\"', open('$ROOT/src/icechannel/__init__.py').read()).group(1))")"
NAME="ice-channel-${VERSION}-py${PYVER/./}"
OUT="$ROOT/bundle/$NAME"
rm -rf "$OUT"
mkdir -p "$OUT/wheels" "$OUT/models" "$OUT/config" "$OUT/calibration"

case "$PLATFORM" in
  x86_64)  PLATS=(--platform manylinux2014_x86_64 --platform manylinux_2_27_x86_64 --platform manylinux_2_28_x86_64) ;;
  *)       PLATS=(--platform "$PLATFORM") ;;
esac

# зависимости onnxruntime для инференса не нужны (--no-deps, -70 МБ)
python3 -m pip download --quiet --only-binary=:all: --no-deps "${PLATS[@]}" \
    --python-version "$PYVER" --implementation cp -d "$OUT/wheels" \
    "numpy>=1.19" "onnxruntime>=1.10" "Pillow>=8.0" "PyYAML>=5.1"

python3 -m pip wheel --quiet --no-deps --no-build-isolation -w "$OUT/wheels" "$ROOT" 2>/dev/null \
  || python3 -m pip wheel --quiet --no-deps -w "$OUT/wheels" "$ROOT"

cp "$ROOT/packaging/install.sh" "$ROOT/packaging/ice-channel.service" "$OUT/"
cp "$ROOT/config/config.example.yaml" "$OUT/config/"
cp "$ROOT"/calibration/*.yaml "$OUT/calibration/"
cp "$ROOT/README.md" "$OUT/"
if ls "$ROOT"/models/*.onnx >/dev/null 2>&1; then
  cp "$ROOT"/models/*.onnx "$ROOT"/models/*.json "$OUT/models/" 2>/dev/null || true
else
  echo "ВНИМАНИЕ: в models/ нет model.onnx — положите модель в бандл вручную (training/export_onnx.py)"
fi
chmod +x "$OUT/install.sh"

tar -C "$ROOT/bundle" -czf "$ROOT/bundle/$NAME.tar.gz" "$NAME"
echo
ls -la "$OUT/wheels" "$OUT/models"
echo "Бандл: bundle/$NAME.tar.gz ($(du -h "$ROOT/bundle/$NAME.tar.gz" | cut -f1))"
