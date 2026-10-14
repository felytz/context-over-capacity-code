"""
_lib_plot._traj_html
====================
Escritor del HTML interactivo de trayectorias distancia–tiempo (compañero de
``trajectories.py``). Diseñado para ser LIVIANO:

- NO embebe los pings crudos ni las ~3,000 trayectorias completas.
- Embebe: la grid de distancias (~250 pts), las bandas de percentiles
  P5/P25/P50/P75/P95 (leve y sin outliers), las medianas de los clusters
  (C y D), una MUESTRA de ≤60 viajes decimada (~120 pts c/u), las pausas
  detectadas (con lat/lon) y el histograma de duraciones.
  Total típico: ~150–300 KB de HTML autocontenido, sin CDN (funciona offline).

Interactividad (vanilla JS + canvas):
- pestañas Filtro leve / Sin outliers / Clusters / Forma,
- toggle de viajes individuales (la muestra),
- crosshair con lectura de percentiles/medianas en el km del mouse,
- pausas como líneas verticales: hover/click → tarjeta con minutos extra,
  % de viajes que pausan y link a Google Maps,
- histograma de duraciones con la cerca de outliers marcada.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

__all__ = ["build_html_payload", "write_trajectories_html"]


def _dec_idx(n: int, max_pts: int = 130) -> np.ndarray:
    """Índices decimados (siempre incluyen el último punto)."""
    step = max(1, n // max_pts)
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _band(T: np.ndarray) -> Dict[str, list]:
    return {f"p{q}": np.round(np.percentile(T, q, axis=0), 2).tolist()
            for q in (5, 25, 50, 75, 95)}


def build_html_payload(*, leg: str, suffix: str, x_km: np.ndarray,
                       T_min: np.ndarray, T_usual: np.ndarray,
                       labels, order, clusters_info: Dict[str, dict],
                       labels_s, order_s, shape_info: Dict[str, dict],
                       bumps: List[dict], lo: float, hi: float, n_raw: int,
                       cluster_style: Dict[str, str],
                       shape_style: Dict[str, str],
                       n_samples: int = 60, seed: int = 0) -> dict:
    """Arma el dict compacto que se embebe como JSON en el HTML."""
    n = len(x_km)
    payload: dict = {
        "leg": leg, "suffix": suffix,
        "route_km": round(float(x_km[-1]), 2),
        "x": np.round(x_km, 3).tolist(),
        "n_raw": int(n_raw), "n_leve": int(T_min.shape[0]),
        "n_usual": int(T_usual.shape[0]),
        "n_out": int(T_min.shape[0] - T_usual.shape[0]),
        "fence": [round(float(max(lo, 0)), 1), round(float(hi), 1)],
        "median_total": round(float(np.median(T_usual[:, -1])), 1),
        "bands": {"leve": _band(T_min), "usual": _band(T_usual)},
        "bumps": bumps,
    }

    # medianas de clusters C (min) y D (%)
    clusters = []
    if labels is not None and clusters_info:
        for rank, j in enumerate(order):
            name = list(clusters_info)[rank]
            Tc = T_usual[labels == j]
            clusters.append({
                "name": name, "color": cluster_style.get(name, "#888"),
                "n": int(len(Tc)),
                "median_min": clusters_info[name]["median_min"],
                "y": np.round(np.percentile(Tc, 50, axis=0), 2).tolist()})
    payload["clusters"] = clusters

    shapes = []
    if labels_s is not None and shape_info:
        T_norm = T_usual / T_usual[:, -1:] * 100.0
        for rank, j in enumerate(order_s):
            name = list(shape_info)[rank]
            Tn = T_norm[labels_s == j]
            shapes.append({
                "name": name, "color": shape_style.get(name, "#888"),
                "n": int(len(Tn)),
                "pct_first_half": shape_info[name]["pct_time_first_half"],
                "y": np.round(np.percentile(Tn, 50, axis=0), 2).tolist()})
    payload["shapes"] = shapes

    # muestra decimada de viajes (representativa del abanico de duraciones)
    idx = _dec_idx(n)
    ns = min(n_samples, len(T_usual))
    by_dur = np.argsort(T_usual[:, -1])
    pick = by_dur[np.linspace(0, len(T_usual) - 1, ns).astype(int)]
    rank_of = {int(j): r for r, j in enumerate(order)} if labels is not None else {}
    rank_of_s = {int(j): r for r, j in enumerate(order_s)} if labels_s is not None else {}
    payload["sx"] = np.round(x_km[idx], 3).tolist()
    payload["samples"] = [
        {"y": np.round(T_usual[i][idx], 1).tolist(),
         "c": rank_of.get(int(labels[i]), -1) if labels is not None else -1,
         "s": rank_of_s.get(int(labels_s[i]), -1) if labels_s is not None else -1}
        for i in pick]

    # histograma de duraciones (sobre el filtro leve; la cerca marca outliers)
    counts, edges = np.histogram(T_min[:, -1], bins=28)
    payload["hist"] = {"counts": counts.tolist(),
                       "edges": np.round(edges, 1).tolist()}
    return payload


# ────────────────────────────────────────────────────────────────────
# Plantilla (dark, autocontenida, sin dependencias externas)
# ────────────────────────────────────────────────────────────────────

_TEMPLATE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trayectorias __LEG__ · __SUFFIX__</title>
<style>
  :root{--bg:#1b1b1b;--card:#262626;--line:#333;--tx:#ddd;--mut:#888;--acc:#0275d8}
  body{font-family:-apple-system,Segoe UI,sans-serif;background:var(--bg);
       color:var(--tx);margin:0;padding:20px;max-width:1200px;margin:auto}
  h1{color:#fff;margin:0 0 2px;font-size:1.35rem}
  h2{color:var(--mut);font-size:.9rem;font-weight:normal;margin:0 0 14px}
  .stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}
  .stat{background:#1f1f1f;padding:8px 14px;border-radius:5px;
        border-left:3px solid var(--acc);min-width:110px}
  .stat .v{font-size:19px;font-weight:bold;color:#fff}
  .stat .l{font-size:10.5px;color:var(--mut);text-transform:uppercase}
  .card{background:var(--card);border-radius:6px;padding:14px;
        border:1px solid var(--line);margin-bottom:14px}
  .tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;align-items:center}
  .tab{background:#1f1f1f;color:var(--tx);border:1px solid var(--line);
       border-radius:4px;padding:6px 12px;cursor:pointer;font-size:13px}
  .tab.on{background:var(--acc);color:#fff;border-color:var(--acc)}
  label.chk{font-size:12.5px;color:var(--mut);margin-left:auto;cursor:pointer;
            user-select:none}
  canvas{width:100%;display:block}
  #tip{position:fixed;display:none;background:#111;border:1px solid #555;
       border-radius:5px;padding:8px 11px;font-size:12.5px;pointer-events:none;
       z-index:10;max-width:290px;line-height:1.5}
  #tip.pin{pointer-events:auto}
  #tip a{color:#6bb3f0}
  .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mut);
          margin-top:8px}
  .legend .sw{display:inline-block;width:14px;height:3px;border-radius:2px;
              vertical-align:middle;margin-right:5px}
  .bumpnote{font-size:12.5px;color:#e0a9a0;margin-top:8px;line-height:1.6}
  .bumpnote a{color:#6bb3f0}
  .foot{color:#666;font-size:11.5px;margin-top:6px}
</style>
</head>
<body>
<h1>Trayectorias distancia–tiempo — leg __LEG__</h1>
<h2>ventana __SUFFIX__ · 0 → __KM__ km · generado por _lib_plot.trajectories</h2>
<div class="stats" id="stats"></div>

<div class="card">
  <div class="tabs" id="tabs">
    <button class="tab on" data-m="leve">Filtro leve</button>
    <button class="tab" data-m="usual">Sin outliers</button>
    <button class="tab" data-m="clusters">Clusters</button>
    <button class="tab" data-m="forma">Forma</button>
    <label class="chk"><input type="checkbox" id="chkS" checked>
      viajes individuales (muestra de <span id="nS"></span>)</label>
  </div>
  <canvas id="main" height="430"></canvas>
  <div class="legend" id="legend"></div>
  <div class="bumpnote" id="bumps"></div>
</div>

<div class="card">
  <b style="font-size:13px">Histograma de duraciones totales (filtro leve)</b>
  <canvas id="hist" height="150"></canvas>
  <div class="foot">La zona sombreada marca la cerca de outliers
    [<span id="f0"></span>, <span id="f1"></span>] min (Tukey 1.5·IQR).</div>
</div>
<div id="tip"></div>

<script>
const D = __DATA__;
const GRAY="#9aa0a6", BUMP="#e74c3c";
let mode="leve", showS=true, pinned=null;

// ── stats chips ──────────────────────────────────────────────────
const chips=[["viajes (crudo)",D.n_raw],["filtro leve",D.n_leve],
  ["sin outliers",D.n_usual],["outliers",D.n_out],
  ["mediana total",D.median_total+" min"],["ruta",D.route_km+" km"],
  ["pausas usuales",D.bumps.length]];
document.getElementById("stats").innerHTML=chips.map(c=>
  `<div class="stat"><div class="v">${typeof c[1]==="number"?c[1].toLocaleString():c[1]}</div><div class="l">${c[0]}</div></div>`).join("");
document.getElementById("nS").textContent=D.samples.length;
document.getElementById("f0").textContent=D.fence[0];
document.getElementById("f1").textContent=D.fence[1];

// ── helpers ──────────────────────────────────────────────────────
function setupCanvas(cv){
  // La altura LOGICA se guarda una sola vez en dataset: cv.height=... pisa el
  // atributo "height", y releerlo cada redraw la multiplicaria por dpr en cada
  // mousemove (el canvas crecia sin fin con escalado de pantalla >100%).
  if(!cv.dataset.h)cv.dataset.h=cv.getAttribute("height");
  const dpr=window.devicePixelRatio||1, w=cv.clientWidth,
        h=parseInt(cv.dataset.h);
  cv.width=w*dpr; cv.height=h*dpr; cv.style.height=h+"px";
  const g=cv.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0);
  return [g,w,h];
}
function interp(xs,ys,x){            // xs creciente
  if(x<=xs[0])return ys[0]; if(x>=xs[xs.length-1])return ys[ys.length-1];
  let i=1; while(xs[i]<x)i++;
  const f=(x-xs[i-1])/(xs[i]-xs[i-1]); return ys[i-1]+f*(ys[i]-ys[i-1]);
}
function hexA(hex,a){
  const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),
        b=parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

// ── gráfica principal ────────────────────────────────────────────
const cv=document.getElementById("main"), tip=document.getElementById("tip");
const M={l:52,r:14,t:10,b:34};
let g,W,H,mouse=null;

function yMax(){
  if(mode==="forma")return 102;
  const b=D.bands[mode==="usual"||mode==="clusters"?"usual":"leve"];
  return Math.max(...b.p95)*1.08;
}
function px(x){return M.l+(x/D.route_km)*(W-M.l-M.r);}
function py(y,ym){return H-M.b-(y/ym)*(H-M.t-M.b);}

function drawCurve(xs,ys,color,lw,alpha,ym){
  g.beginPath();
  for(let i=0;i<xs.length;i++){
    const X=px(xs[i]),Y=py(ys[i],ym);
    i?g.lineTo(X,Y):g.moveTo(X,Y);
  }
  g.strokeStyle=alpha<1?hexA(color,alpha):color; g.lineWidth=lw; g.stroke();
}
function fillBand(xs,yTop,yBot,color,alpha,ym){
  g.beginPath();
  for(let i=0;i<xs.length;i++){const X=px(xs[i]);
    i?g.lineTo(X,py(yTop[i],ym)):g.moveTo(X,py(yTop[i],ym));}
  for(let i=xs.length-1;i>=0;i--)g.lineTo(px(xs[i]),py(yBot[i],ym));
  g.closePath(); g.fillStyle=hexA(color,alpha); g.fill();
}
function axes(ym,yLab){
  g.strokeStyle="#3a3a3a"; g.fillStyle="#9aa"; g.font="11px sans-serif";
  const xt=Math.max(1,Math.round(D.route_km/12));
  for(let x=0;x<=D.route_km;x+=xt){
    const X=px(x); g.beginPath(); g.moveTo(X,M.t); g.lineTo(X,H-M.b);
    g.lineWidth=.5; g.stroke();
    g.textAlign="center"; g.fillText(x,X,H-M.b+16);
  }
  const step=ym>150?30:(ym>60?20:(ym>25?10:5));
  for(let y=0;y<=ym;y+=step){
    const Y=py(y,ym); g.beginPath(); g.moveTo(M.l,Y); g.lineTo(W-M.r,Y);
    g.lineWidth=.5; g.stroke();
    g.textAlign="right"; g.fillText(y,M.l-6,Y+4);
  }
  g.textAlign="center";
  g.fillText("distancia recorrida (km)",(M.l+W-M.r)/2,H-4);
  g.save(); g.translate(12,(M.t+H-M.b)/2); g.rotate(-Math.PI/2);
  g.fillText(yLab,0,0); g.restore();
}
function normSample(y){const f=100/y[y.length-1];return y.map(v=>v*f);}

function draw(){
  [g,W,H]=setupCanvas(cv);
  const ym=yMax(), yLab=mode==="forma"?"% del tiempo total":"tiempo (min)";
  axes(ym,yLab);
  const legend=[];

  if(showS){                                    // muestra de viajes
    for(const s of D.samples){
      let col=GRAY, y=s.y;
      if(mode==="clusters"&&s.c>=0&&D.clusters[s.c])col=D.clusters[s.c].color;
      if(mode==="forma"){y=normSample(s.y);
        if(s.s>=0&&D.shapes[s.s])col=D.shapes[s.s].color;}
      drawCurve(D.sx,y,col,0.7,0.30,ym);
    }
  }
  if(mode==="leve"||mode==="usual"){
    const b=D.bands[mode];
    fillBand(D.x,b.p95,b.p5,"#0275d8",0.10,ym);
    fillBand(D.x,b.p75,b.p25,"#0275d8",0.16,ym);
    drawCurve(D.x,b.p5,"#0275d8",2,1,ym);
    drawCurve(D.x,b.p50,"#f5f5f5",2.4,1,ym);
    drawCurve(D.x,b.p95,"#d9534f",2,1,ym);
    legend.push(["#0275d8","P5 rápido"],["#f5f5f5","mediana"],
                ["#d9534f","P95 lento"],["rgba(2,117,216,.25)","banda P25–P75"]);
  }else if(mode==="clusters"){
    for(const c of D.clusters){drawCurve(D.x,c.y,c.color,2.6,1,ym);
      legend.push([c.color,`${c.name} — ${c.n.toLocaleString()} viajes · mediana ${c.median_min} min`]);}
  }else{
    g.setLineDash([5,4]);
    drawCurve([0,D.route_km],[0,100],"#bbb",1,1,ym); g.setLineDash([]);
    for(const s of D.shapes){drawCurve(D.x,s.y,s.color,2.6,1,ym);
      legend.push([s.color,`${s.name} — ${s.n.toLocaleString()} viajes · ${s.pct_first_half}% del tiempo en la 1ª mitad`]);}
    legend.push(["#bbb","proporcional a la distancia"]);
  }

  for(const b of D.bumps){                       // pausas
    const X=px(b.km);
    g.setLineDash([5,4]); g.strokeStyle=BUMP; g.lineWidth=1.6;
    g.beginPath(); g.moveTo(X,M.t); g.lineTo(X,H-M.b); g.stroke();
    g.setLineDash([]);
    g.fillStyle=BUMP; g.font="12px sans-serif"; g.textAlign="center";
    g.fillText("⏸",X,M.t+12);
  }
  if(mouse&&!pinned){                            // crosshair
    const X=mouse.x;
    g.strokeStyle="#777"; g.lineWidth=.8; g.setLineDash([3,3]);
    g.beginPath(); g.moveTo(X,M.t); g.lineTo(X,H-M.b); g.stroke();
    g.setLineDash([]);
  }
  document.getElementById("legend").innerHTML=legend.map(l=>
    `<span><span class="sw" style="background:${l[0]}"></span>${l[1]}</span>`).join("");
}

// ── tooltip (crosshair + pausas con click para fijar) ────────────
function bumpAt(xpx){
  for(const b of D.bumps)if(Math.abs(px(b.km)-xpx)<7)return b;
  return null;
}
function bumpHTML(b){
  const maps=b.lat!=null?`<br><a href="https://maps.google.com/?q=${b.lat},${b.lon}" target="_blank">📍 ${b.lat}, ${b.lon} — ver en Google Maps</a>`:"";
  return `<b style="color:${BUMP}">⏸ pausa usual · km ${b.km.toFixed(1)}</b><br>`+
    `tramo ${b.from_km.toFixed(1)}–${b.to_km.toFixed(1)} km<br>`+
    `+${b.extra_min_median_trip} min extra (mediana por viaje)<br>`+
    `${b.pct_trips_pausing}% de los viajes pausan aquí${maps}`;
}
function showTip(html,cx,cy,pin){
  tip.innerHTML=html; tip.style.display="block";
  tip.classList.toggle("pin",!!pin);
  tip.style.left=Math.min(cx+14,window.innerWidth-310)+"px";
  tip.style.top=(cy+14)+"px";
}
cv.addEventListener("mousemove",e=>{
  if(pinned)return;
  const r=cv.getBoundingClientRect(), x=e.clientX-r.left;
  if(x<M.l||x>W-M.r){tip.style.display="none";mouse=null;draw();return;}
  mouse={x}; draw();
  const km=(x-M.l)/(W-M.l-M.r)*D.route_km, b=bumpAt(x);
  if(b){showTip(bumpHTML(b),e.clientX,e.clientY,false);return;}
  let html=`<b>km ${km.toFixed(1)}</b><br>`;
  if(mode==="leve"||mode==="usual"){
    const bd=D.bands[mode];
    html+=`mediana ${interp(D.x,bd.p50,km).toFixed(1)} min<br>`+
          `P5 ${interp(D.x,bd.p5,km).toFixed(1)} · P95 ${interp(D.x,bd.p95,km).toFixed(1)} min`;
  }else if(mode==="clusters"){
    html+=D.clusters.map(c=>`<span style="color:${c.color}">●</span> ${c.name}: ${interp(D.x,c.y,km).toFixed(1)} min`).join("<br>");
  }else{
    html+=D.shapes.map(s=>`<span style="color:${s.color}">●</span> ${s.name}: ${interp(D.x,s.y,km).toFixed(1)}%`).join("<br>");
  }
  showTip(html,e.clientX,e.clientY,false);
});
cv.addEventListener("mouseleave",()=>{if(!pinned){tip.style.display="none";mouse=null;draw();}});
cv.addEventListener("click",e=>{
  const r=cv.getBoundingClientRect(), b=bumpAt(e.clientX-r.left);
  if(pinned){pinned=null;tip.style.display="none";draw();return;}
  if(b){pinned=b;showTip(bumpHTML(b),e.clientX,e.clientY,true);}
});

// ── nota de pausas bajo la gráfica ───────────────────────────────
document.getElementById("bumps").innerHTML=D.bumps.length?D.bumps.map(b=>{
  const maps=b.lat!=null?` · <a href="https://maps.google.com/?q=${b.lat},${b.lon}" target="_blank">📍 ${b.lat}, ${b.lon}</a>`:"";
  return `⏸ <b>km ${b.km.toFixed(1)}</b> (${b.from_km.toFixed(1)}–${b.to_km.toFixed(1)}): `+
    `+${b.extra_min_median_trip} min mediana por viaje · ${b.pct_trips_pausing}% de viajes pausan${maps}`;
}).join("<br>"):"Sin pausas usuales detectadas a media ruta.";

// ── tabs + toggle ────────────────────────────────────────────────
document.getElementById("tabs").addEventListener("click",e=>{
  const t=e.target.closest(".tab"); if(!t)return;
  document.querySelectorAll(".tab").forEach(b=>b.classList.remove("on"));
  t.classList.add("on"); mode=t.dataset.m; pinned=null;
  tip.style.display="none"; draw();
});
document.getElementById("chkS").addEventListener("change",e=>{
  showS=e.target.checked; draw();});

// ── histograma de duraciones ─────────────────────────────────────
function drawHist(){
  const hcv=document.getElementById("hist");
  const [hg,w,h]=setupCanvas(hcv);
  const cts=D.hist.counts, ed=D.hist.edges,
        mx=Math.max(...cts), m={l:44,r:10,t:8,b:26};
  const X=v=>m.l+(v-ed[0])/(ed[ed.length-1]-ed[0])*(w-m.l-m.r);
  // cerca de outliers (zona usual sombreada)
  hg.fillStyle="rgba(92,184,92,.08)";
  hg.fillRect(X(D.fence[0]),m.t,X(D.fence[1])-X(D.fence[0]),h-m.t-m.b);
  for(let i=0;i<cts.length;i++){
    const x0=X(ed[i]),x1=X(ed[i+1]),
          out=ed[i+1]<=D.fence[0]||ed[i]>=D.fence[1];
    hg.fillStyle=out?hexA("#d9534f",.75):hexA("#0275d8",.8);
    const bh=(cts[i]/mx)*(h-m.t-m.b);
    hg.fillRect(x0+.5,h-m.b-bh,Math.max(1,x1-x0-1),bh);
  }
  hg.fillStyle="#9aa"; hg.font="11px sans-serif"; hg.textAlign="center";
  const tks=6;
  for(let i=0;i<=tks;i++){
    const v=ed[0]+(ed[ed.length-1]-ed[0])*i/tks;
    hg.fillText(Math.round(v),X(v),h-8);
  }
  hg.save(); hg.translate(11,(m.t+h-m.b)/2); hg.rotate(-Math.PI/2);
  hg.fillText("viajes",0,0); hg.restore();
}

window.addEventListener("resize",()=>{draw();drawHist();});
draw(); drawHist();
</script>
</body>
</html>
"""


def write_trajectories_html(payload: dict, path: Path) -> Path:
    """Escribe el HTML autocontenido; devuelve la ruta."""
    html = (_TEMPLATE
            .replace("__LEG__", payload["leg"])
            .replace("__SUFFIX__", payload["suffix"])
            .replace("__KM__", str(payload["route_km"]))
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False,
                                            separators=(",", ":"))))
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
