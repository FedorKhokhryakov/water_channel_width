#!/bin/sh
# Установка на судне без интернета и pip: wheel-архивы распаковываются в $PREFIX/lib.
#   sudo ./install.sh [--systemd] [--prefix /opt/ice-channel]
set -eu

PREFIX=/opt/ice-channel
SYSTEMD=0
USER_NAME="${SUDO_USER:-$(id -un)}"
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --systemd) SYSTEMD=1; shift ;;
    --user) USER_NAME="$2"; shift 2 ;;
    *) echo "неизвестный параметр: $1"; exit 2 ;;
  esac
done
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-$(command -v python3)}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 7), sys.version' \
  || { echo "Нужен Python >= 3.7"; exit 1; }
echo "Python: $("$PY" --version 2>&1) ($PY), установка в $PREFIX"

mkdir -p "$PREFIX/lib" "$PREFIX/bin" "$PREFIX/models" "$PREFIX/calibration" "$PREFIX/config"
rm -rf "$PREFIX/lib.new" && mkdir "$PREFIX/lib.new"
"$PY" - "$HERE/wheels" "$PREFIX/lib.new" <<'EOF'
import os, shutil, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
for name in sorted(os.listdir(src)):
    if name.endswith(".whl"):
        zipfile.ZipFile(os.path.join(src, name)).extractall(dst)
        print("  +", name)
# лишнее для работы
junk = ["numpy/tests", "numpy/f2py", "numpy/distutils", "numpy/testing/tests", "numpy/_pyinstaller",
        "onnxruntime/transformers", "onnxruntime/quantization", "onnxruntime/tools",
        "onnxruntime/datasets", "onnxruntime/backend"]
for root, dirs, _ in os.walk(dst):
    for d in list(dirs):
        if d in ("tests", "__pycache__"):
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
for j in junk:
    shutil.rmtree(os.path.join(dst, j), ignore_errors=True)
for root, _, files in os.walk(dst):
    for f in files:
        if ".so" in f:
            os.chmod(os.path.join(root, f), 0o755)
EOF
rm -rf "$PREFIX/lib.old"; [ -d "$PREFIX/lib" ] && mv "$PREFIX/lib" "$PREFIX/lib.old"
mv "$PREFIX/lib.new" "$PREFIX/lib" && rm -rf "$PREFIX/lib.old"

cp -f "$HERE"/models/* "$PREFIX/models/" 2>/dev/null || echo "ВНИМАНИЕ: в бандле нет модели (models/)"
for f in "$HERE"/calibration/*.yaml; do
  [ -e "$PREFIX/calibration/$(basename "$f")" ] || cp "$f" "$PREFIX/calibration/"
done
cp -f "$HERE/config/config.example.yaml" "$PREFIX/config/"
if [ ! -e "$PREFIX/config/config.yaml" ]; then
  sed -e 's#\.\./models/#'"$PREFIX"'/models/#' -e 's#\.\./calibration/#'"$PREFIX"'/calibration/#' \
      "$PREFIX/config/config.example.yaml" > "$PREFIX/config/config.yaml"
  echo "Создан $PREFIX/config/config.yaml — проверьте paths.inbox и paths.output"
fi

cat > "$PREFIX/bin/ice-channel" <<EOF
#!/bin/sh
PYTHONPATH="$PREFIX/lib\${PYTHONPATH:+:\$PYTHONPATH}" exec "$PY" -s -m icechannel "\$@"
EOF
chmod +x "$PREFIX/bin/ice-channel"

if [ "$SYSTEMD" = 1 ]; then
  sed -e "s#@PREFIX@#$PREFIX#g" -e "s#@USER@#$USER_NAME#g" "$HERE/ice-channel.service" \
      > /etc/systemd/system/ice-channel.service
  systemctl daemon-reload
  systemctl enable ice-channel.service
  echo "Служба установлена: systemctl start ice-channel; journalctl -u ice-channel -f"
fi

echo "Размер: $(du -sh "$PREFIX" | cut -f1)"
"$PREFIX/bin/ice-channel" check -c "$PREFIX/config/config.yaml" || true
echo "Запуск вручную: $PREFIX/bin/ice-channel run -c $PREFIX/config/config.yaml"
