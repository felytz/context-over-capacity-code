"""Mapa estático de la zona distinta de una familia con variante (variantes_ruta.py): calles
(Esri) en gris y las dos trazas con flechas de sentido, la principal en azul y la alterna en
naranja, cada una corrida unos píxeles a su derecha para que se vean las dos cuando van por la
misma calle al revés (1101, 1401). En grueso, la parte exclusiva de cada una; los puntos, sus
paradas exclusivas. No usa crudo.

    python analisis/variantes_mapa.py 1101 [--margen 250] [--salida PNG]

Va a reports/variantes/<familia>/zona.png.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "analisis"))
import mapas_traza as mt  # noqa: E402
import variantes_ruta as vr  # noqa: E402
import zona_mapa as zm  # noqa: E402

COLORES = ("#1f5fbf", "#e8590c")
DESPLAZA_PX = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("familia", choices=list(vr.FAMILIAS))
    ap.add_argument("--margen", type=float, default=250.0, help="m alrededor de la zona")
    ap.add_argument("--ancho-px", type=int, default=1500, help="ancho máximo del fondo (elige el zoom)")
    ap.add_argument("--salida")
    a = ap.parse_args()
    variantes = vr.FAMILIAS[a.familia]["variantes"]
    T = vr.geometria(variantes, vr.RAW)
    zona = np.vstack([T[v]["q"][T[v]["excl"]] for v in variantes])
    lo_x, hi_x = zona[:, 0].min() - a.margen, zona[:, 0].max() + a.margen
    lo_y, hi_y = zona[:, 1].min() - a.margen, zona[:, 1].max() + a.margen
    lat0, lat1, lon0, lon1 = lo_y / mt.KY, hi_y / mt.KY, lo_x / mt.KX, hi_x / mt.KX
    z = 18
    while z > 12:
        px0, _ = zm.a_px(lat0, lon0, z)
        px1, _ = zm.a_px(lat1, lon1, z)
        if px1 - px0 <= a.ancho_px:
            break
        z -= 1
    px0, py1 = zm.a_px(lat0, lon0, z)
    px1, py0 = zm.a_px(lat1, lon1, z)
    tx0, tx1, ty0, ty1 = int(px0 // 256), int(px1 // 256), int(py0 // 256), int(py1 // 256)
    fondo = Image.new("L", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256), 255)
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            fondo.paste(zm.tesela(z, tx, ty), ((tx - tx0) * 256, (ty - ty0) * 256))
    fondo = Image.eval(fondo, lambda v: int(150 + v * 0.42))
    ox, oy = tx0 * 256, ty0 * 256

    def P(xm, ym):
        X, Y = zm.a_px(np.asarray(ym) / mt.KY, np.asarray(xm) / mt.KX, z)
        return X - ox, Y - oy

    fig, ax = plt.subplots(figsize=(12, 12 * (py1 - py0) / max(px1 - px0, 1)), dpi=110)
    ax.imshow(fondo, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
    g = mt.archivos_geometria(vr.RAW)
    stops = {s["code"]: s for s in json.load(open(g["stops"], encoding="utf-8"))}
    rutas = json.load(open(g["stops_routes"], encoding="utf-8"))
    for k, (v, o) in enumerate((variantes, variantes[::-1])):
        q, rum, s = T[v]["q"], T[v]["rumbo"], T[v]["s"]
        X, Y = P(q[:, 0], q[:, 1])
        ang = np.radians(rum)
        X, Y = X - DESPLAZA_PX * np.cos(ang) * -1, Y + DESPLAZA_PX * np.sin(ang)      # a la derecha del sentido
        dentro = (q[:, 0] >= lo_x) & (q[:, 0] <= hi_x) & (q[:, 1] >= lo_y) & (q[:, 1] <= hi_y)
        for grupo in np.split(np.flatnonzero(dentro), np.flatnonzero(np.diff(np.flatnonzero(dentro)) > 1) + 1):
            if len(grupo) > 1:
                ax.plot(X[grupo], Y[grupo], color=COLORES[k], lw=1.4, alpha=.8)
        largo = sum(len(r) for r in T[v]["tramos"]) * mt.PASO_M / 1000
        nom = vr.NOMBRE.get(v, v)
        for n, r in enumerate(T[v]["tramos"]):
            ax.plot(X[r], Y[r], color=COLORES[k], lw=4.5, alpha=.55,
                    label=f"{nom}: sólo suya, km {s[T[v]['tramos'][0][0]] / 1000:.1f}–… ({largo:.1f} km)" if n == 0 else None)
        for i in np.flatnonzero(dentro)[3::12]:
            ax.annotate("", xy=(X[i] + 10 * math.sin(ang[i]), Y[i] - 10 * math.cos(ang[i])), xytext=(X[i], Y[i]),
                        arrowprops=dict(arrowstyle="-|>", color=COLORES[k], lw=1.3))
        for i in np.flatnonzero(dentro)[::100]:
            ax.annotate(f"{s[i] / 1000:.0f}", (X[i], Y[i]), fontsize=7, color=COLORES[k], xytext=(5, -9 if k else 5),
                        textcoords="offset points", bbox=dict(boxstyle="round,pad=.1", fc="white", ec="none", alpha=.8))
        otras = {e["stop_code"] for e in rutas.get(o, [])}
        propias = [stops[e["stop_code"]] for e in rutas.get(v, []) if e["stop_code"] not in otras and e["stop_code"] in stops]
        if propias:
            sx, sy = P(np.array([p["longitude"] for p in propias], float) * mt.KX,
                       np.array([p["latitude"] for p in propias], float) * mt.KY)
            ax.scatter(sx, sy, s=22, color=COLORES[k], edgecolor="white", lw=.8, zorder=5,
                       label=f"{nom}: {len(propias)} paradas que la otra no tiene")
    alt = vr.NOMBRE.get(variantes[1], variantes[1])
    ax.set_title(f"Familia {a.familia}: {variantes[0]} (azul) / {alt}{f' = {variantes[1]}' if alt != variantes[1] else ''} (naranja) — "
                 f"al revés por las mismas calles: {T[variantes[0]]['contrario_m'] / 1000:.1f} km", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, framealpha=.9)
    ax.set_xlim(*P(np.array([lo_x, hi_x]), np.array([lo_y, lo_y]))[0])
    ax.set_ylim(P(np.array([lo_x, lo_x]), np.array([lo_y, hi_y]))[1])
    ax.set_axis_off()
    ax.text(.995, .003, "Calles: Esri", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#555")
    salida = Path(a.salida) if a.salida else vr.REPORTES / a.familia / "zona.png"
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, bbox_inches="tight", pad_inches=.05)
    print("→", salida)


if __name__ == "__main__":
    main()
