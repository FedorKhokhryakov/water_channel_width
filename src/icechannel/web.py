"""Веб-страница результатов на стандартной библиотеке (только чтение, 127.0.0.1)."""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List
from urllib.parse import parse_qs, unquote, urlparse

log = logging.getLogger("icechannel")

PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ширина канала</title><style>
:root{--bg:#f6f7f9;--fg:#1d2430;--mut:#6b7380;--card:#fff;--line:#2b8fd6;--ok:#2e9e57;--sus:#d98a00;--bad:#d64545}
@media (prefers-color-scheme:dark){:root{--bg:#14181f;--fg:#e6e9ee;--mut:#98a1ad;--card:#1d232c;--line:#59b0ef}}
body{margin:0;font:14px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
main{max-width:1200px;margin:auto;padding:16px}.row{display:flex;gap:16px;flex-wrap:wrap}
.card{background:var(--card);border-radius:10px;padding:14px;flex:1;min-width:260px}
.big{font-size:40px;font-weight:600}.mut{color:var(--mut)}img{max-width:100%;border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:3px 6px;border-bottom:1px solid #8883;text-align:right}
.ok{color:var(--ok)}.suspect{color:var(--sus)}.fail{color:var(--bad)}canvas{width:100%;height:260px}
pre{white-space:pre-wrap;font-size:11px;max-height:300px;overflow:auto}a{color:var(--line)}
</style></head><body><main>
<h2>Ширина ледового канала за кормой</h2>
<div class="row"><div class="card"><div class="mut">Последний кадр</div><div class="big" id="w">—</div>
<div id="info" class="mut"></div></div><div class="card"><div class="mut">Сервис</div><div id="svc"></div></div></div>
<div class="row" style="margin-top:16px"><div class="card"><div class="mut">Ширина, м
(точки — кадры, линия — сглаженная) · <select id="hours"><option>6</option><option selected>24</option>
<option>72</option><option>168</option></select> ч</div><canvas id="ch"></canvas></div></div>
<div class="row" style="margin-top:16px"><div class="card"><img id="ov" alt="нет overlay"></div></div>
<div class="row" style="margin-top:16px"><div class="card"><div class="mut">Последние кадры ·
<a href="/files/">CSV по дням</a></div><table id="tb"></table></div></div>
<div class="row" style="margin-top:16px"><div class="card"><div class="mut">Лог</div><pre id="log"></pre></div></div>
</main><script>
const $=id=>document.getElementById(id);
async function j(u){const r=await fetch(u);return r.json()}
function chart(rows){const c=$('ch'),dpr=devicePixelRatio||1;c.width=c.clientWidth*dpr;c.height=260*dpr;
const g=c.getContext('2d');g.scale(dpr,dpr);const W=c.clientWidth,H=260,p=36;g.clearRect(0,0,W,H);
const pts=rows.filter(r=>r.width_m!=='').map(r=>({t:Date.parse(r.time.replace(' ','T')),w:+r.width_m,s:r.width_smooth_m===''?null:+r.width_smooth_m,st:r.status}));
if(!pts.length){g.fillStyle='#888';g.fillText('нет данных',W/2-30,H/2);return}
const t0=Math.min(...pts.map(p=>p.t)),t1=Math.max(...pts.map(p=>p.t))+1;
const ws=pts.map(p=>p.w).sort((a,b)=>a-b),lo=Math.max(0,ws[Math.floor(ws.length*.02)]-5),hi=ws[Math.floor(ws.length*.98)]+5;
const X=t=>p+(t-t0)/(t1-t0)*(W-2*p),Y=w=>H-p+( -(w-lo)/(hi-lo))*(H-2*p);
const css=getComputedStyle(document.documentElement);g.strokeStyle='#8884';g.fillStyle=css.getPropertyValue('--mut');
for(let k=0;k<=4;k++){const w=lo+(hi-lo)*k/4,y=Y(w);g.beginPath();g.moveTo(p,y);g.lineTo(W-p,y);g.stroke();g.fillText(w.toFixed(0),4,y+4)}
const col={ok:css.getPropertyValue('--ok'),suspect:css.getPropertyValue('--sus'),fail:css.getPropertyValue('--bad')};
for(const q of pts){g.fillStyle=col[q.st]||'#888';g.fillRect(X(q.t)-1.5,Y(q.w)-1.5,3,3)}
g.strokeStyle=css.getPropertyValue('--line');g.lineWidth=2;g.beginPath();let f=1;
for(const q of pts){if(q.s===null)continue;f?g.moveTo(X(q.t),Y(q.s)):g.lineTo(X(q.t),Y(q.s));f=0}g.stroke()}
async function load(){
 const L=await j('/api/latest');if(L.time){$('w').textContent=(L.width_m===''||L.width_m==null?'—':L.width_m+' м');
 $('w').className='big '+L.status;$('info').textContent=`${L.time} · ${L.status} ${L.flags||L.reason||''} · сглаж. ${L.width_smooth_m??'—'} м · pitch ${L.pitch_deg}° roll ${L.roll_deg}° (${L.horizon_source})`;
 $('ov').src='/overlay/latest?'+Date.now()}
 const S=await j('/api/status');$('svc').innerHTML=S.heartbeat?`пульс ${S.heartbeat}<br>обработано ${S.processed} (ok ${S.ok}, сомнит. ${S.suspect}, нет ${S.fail}, ошибок ${S.errors})<br>свободно ${S.free_mb} МБ · v${S.version}`:'нет данных';
 const R=await j('/api/series?hours='+$('hours').value);chart(R);
 const cols=['time','status','width_m','width_smooth_m','width_near_m','w_100m','w_200m','w_300m','pitch_deg','roll_deg','flags'];
 $('tb').innerHTML='<tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr>'+R.slice(-25).reverse().map(r=>'<tr>'+cols.map(c=>`<td class="${c==='status'?r.status:''}">${r[c]??''}</td>`).join('')+'</tr>').join('');
 $('log').textContent=(await (await fetch('/api/log?n=80')).text())}
$('hours').onchange=load;load();setInterval(load,30000);
</script></body></html>"""


def _read_series(root: Path, hours: float) -> List[dict]:
    since = datetime.now() - timedelta(hours=hours)
    rows: List[dict] = []
    days = int(hours // 24) + 2
    for f in sorted((root / "results").glob("*.csv"))[-days:]:
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    if rows:
        # отсчитываем от последнего кадра, а не от часов сервера
        try:
            last = max(datetime.fromisoformat(r["time"]) for r in rows if r.get("time"))
            since = min(since, last - timedelta(hours=hours))
        except ValueError:
            pass
        rows = [r for r in rows if r.get("time") and datetime.fromisoformat(r["time"]) >= since]
    keep = ("time", "file", "status", "flags", "reason", "width_m", "width_smooth_m", "width_near_m",
            "w_100m", "w_150m", "w_200m", "w_250m", "w_300m", "pitch_deg", "roll_deg", "horizon_source")
    return [{k: r.get(k, "") for k in keep} for r in rows]


def make_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # тише стандартного вывода
            log.debug("web: " + fmt, *args)

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj) -> None:
            self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _file(self, p: Path, ctype: str) -> None:
            try:
                p.resolve().relative_to(root.resolve())
            except ValueError:
                return self._send(b"forbidden", "text/plain", 403)
            if not p.is_file():
                return self._send(b"not found", "text/plain", 404)
            self._send(p.read_bytes(), ctype)

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            path = unquote(u.path)
            if path == "/":
                return self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/latest":
                p = root / "latest.json"
                return self._json(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {})
            if path == "/api/status":
                p = root / "state" / "status.json"
                return self._json(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {})
            if path == "/api/series":
                return self._json(_read_series(root, float(q.get("hours", ["24"])[0])))
            if path == "/api/log":
                p = root / "logs" / "service.log"
                n = int(q.get("n", ["100"])[0])
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:] if p.exists() else []
                return self._send("\n".join(lines).encode("utf-8"), "text/plain; charset=utf-8")
            if path == "/overlay/latest":
                files = sorted((root / "overlays").glob("*/*.jpg"))
                if not files:
                    return self._send(b"", "image/jpeg", 404)
                return self._file(files[-1], "image/jpeg")
            if path == "/files/":
                items = sorted((root / "results").glob("*.csv"), reverse=True)
                html = "<h3>CSV</h3>" + "".join('<a href="/files/results/%s">%s</a><br>' % (f.name, f.name)
                                                for f in items)
                return self._send(html.encode("utf-8"), "text/html; charset=utf-8")
            if path.startswith("/files/"):
                p = root / path[len("/files/"):]
                ctype = "text/csv; charset=utf-8" if p.suffix == ".csv" else "image/jpeg"
                return self._file(p, ctype)
            self._send(b"not found", "text/plain", 404)

    return Handler


def serve(cfg, host: str = "127.0.0.1", port: int = 8585) -> None:
    root = cfg.path("output")
    srv = ThreadingHTTPServer((host, port), make_handler(root))
    print("Веб-страница: http://%s:%d  (данные: %s)" % (host, port, root))
    if host in ("127.0.0.1", "localhost"):
        print("С берега: ssh -L %d:127.0.0.1:%d user@<судно>, затем http://localhost:%d" % (port, port, port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
