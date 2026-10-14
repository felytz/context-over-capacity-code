"""Figure 1 (versión 9k) — la cadena de construcción y el corte de la vuelta.

Panel (a): la cadena de extremo a extremo. Panel (b): cómo se corta un día de un bus en
vueltas, con DATOS REALES (`figure_01b_round_trips.{csv,json}`, que escribe
`extract_figure_01b.py` corriendo el corte del pipeline sobre un bus-día).

Con la vuelta (rama `vuelta`, fase 1b del plan)
-----------------------------------------------
* **El panel (b) ya no es el detector de reversión.** En agosto la Figura 1 fundía la
  cadena con el descubrimiento de terminales (decisión D2 de `DECISIONES.md`: una
  contribución declarada que no podía bajar al Online Resource). Con la vuelta ese
  detector no corta: el corte es el cruce del punto de corte sobre el avance continuo
  del bus. El panel lo dibuja como «km dentro de la vuelta» contra la hora: cada vuelta
  es un diente de 0 al largo del lazo; los pedazos de antes del primer cruce y de
  después del último (y los que deja un salto imposible) no son vueltas. En una ruta de
  dos terminales la espera en la otra se ve como un escalón dentro del diente (P8).
* **Panel (a)**: las etapas 2–3 pasan a ser el avance sobre la traza y el corte en
  vueltas; la 5 y la 6, «por ruta». El corte de secuencias en huecos > 7 min (P90) que
  decía la etapa 1 era de la segmentación vieja (`step_06`), que el pipeline nuevo no
  usa; los tramos se parten ahora en huecos de 20 min (etapa 3).
* **Panel (a), 10-oct (fase 3 del manuscrito)**: la etapa 1 decía «~30.6 %» de quietos y
  un filtro de 50 m/s, que eran de agosto (medido: 27.4 %, `analysis/cleaning_stats`; el
  salto imposible parte el tramo, etapa 3); la 4, un «37-variable contract» que `df_full`
  ya no tiene (22 columnas); la 6, una «post-prediction audit» que ya no existe.

⚠ RESTRICCIÓN DE LA REVISTA. *Public Transport* (Springer) admite como mucho
**174 mm de ancho y 234 mm de alto** (6.85 × 9.21 in). El guard del final lo comprueba.

⚠ Las etiquetas del panel (b) se colocan a mano: matplotlib no evita colisiones y no
avisa de ninguna.

Reproducible con el venv del repo:
    .venv/Scripts/python.exe docs/paper/figures/make_figure_01_merged.py
Salidas: figure_01_merged.{png,pdf} junto a este script.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, FancyBboxPatch

# Okabe-Ito (colorblind-safe), la misma familia que `_lib_style` del repo
C_DATA = "#56B4E9"      # sky blue        — fuentes de datos
C_PROC = "#009E73"      # bluish green    — construcción del viaje
C_FEAT = "#0072B2"      # blue            — features
C_MODEL = "#E69F00"     # orange          — modelos
C_EVAL = "#CC79A7"      # reddish purple  — evaluación
C_PATH = "0.55"
C_PING = "#0072B2"
C_VIN = "#009E73"
C_VOUT = "#D55E00"      # vermillion
C_EVENT = "#E69F00"
C_REJ = "#CC79A7"
EDGE_W = 1.2

FIG_W, FIG_H = 6.85, 9.15          # el máximo de Springer, menos un pelo
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = GridSpec(2, 1, height_ratios=[1.72, 1.0], hspace=0.045,
              left=0.012, right=0.988, top=0.985, bottom=0.012)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])
for ax in (axA, axB):
    ax.set_xlim(0, 100)
    ax.axis("off")
axA.set_ylim(0, 100)
axB.set_ylim(0, 100)

#: (objeto de texto, ancho máximo en unidades de datos) — ver fit_texts()
_FITS: list[tuple[object, float]] = []


def box(x, y, w, h, color, title, lines, title_size=7.6, text_size=6.3):
    """Caja redondeada con título en negrita y cuerpo debajo. (x, y) = centro."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.5,rounding_size=1.4",
        linewidth=EDGE_W, edgecolor=color,
        facecolor=matplotlib.colors.to_rgba(color, 0.13),
        mutation_aspect=1.0,
    )
    axA.add_patch(patch)
    title_y = y + h / 2 - 2.2
    t = axA.text(x, title_y, title, ha="center", va="center",
                 fontsize=title_size, fontweight="bold", color="black")
    b = axA.text(x, title_y - 1.9, "\n".join(lines), ha="center", va="top",
                 fontsize=text_size, color="black", linespacing=1.45)
    _FITS.extend([(t, w - 1.5), (b, w - 1.5)])
    return t


def fit_texts():
    """Encoge el tipo de cualquier texto que se salga de su caja.

    Sin esto el desbordamiento es SILENCIOSO: matplotlib dibuja el texto fuera
    del rectángulo sin avisar, y en el PDF final parece un error de maquetado.
    Heredado de `make_figure_01_pipeline.py`, donde costó un defecto real.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = axA.transData.inverted()
    for t, max_w in _FITS:
        while t.get_fontsize() > 4.8:
            bb = t.get_window_extent(r)
            (x0, _), (x1, _) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
            if (x1 - x0) <= max_w:
                break
            t.set_fontsize(t.get_fontsize() - 0.15)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.1,
                                 color="0.25", shrinkA=2, shrinkB=2))


# ══════════════════ Panel (a) — la cadena de extremo a extremo ══════════════
axA.text(0.5, 99.0, "a", fontsize=12, fontweight="bold", va="top", ha="left")

box(28, 94.0, 46, 8.6, C_DATA, "Raw AVL records",
    ["one ping per vehicle · nominal ~60 s", "Jan–Aug 2024"])
box(76, 94.0, 40, 8.6, C_DATA, "Route geometry & stops",
    ["polylines · stop inventory", "spatial reference only — no schedules, no GTFS"])

box(50, 80.5, 90, 8.2, C_PROC, "1 · Record cleaning and projection",
    ["duplicates and off-hours records removed · stationary records retained (27.4%)",
     "records projected onto the route polyline, coherently in time"])

box(50, 68.0, 90, 8.8, C_PROC, "2 · Continuous progress and one cut point per route",
    ["progress along the polyline summed without resetting at each loop",
     "cut point fixed in advance: a layover, dispatch or terminal location"])

box(50, 55.5, 90, 8.2, C_PROC, "3 · Segmentation into round trips (panel b)",
    ["stretches split at day end, gaps > 20 min, jumps that cannot be driven",
     "round trip: one crossing of the cut point to the next · ≥ 5 pings · no missed crossing"])

box(50, 43.5, 90, 7.4, C_FEAT, "4 · Target and features",
    ["interpolated target on the common 100 m grid along the polyline",
     "historical quantities from the training fold only (leakage-free)"])

# --- etapa 5: contenedor con las tres familias -------------------------------
# El encuadre importa: `delay_hour` va con los baselines y no con los tabulares,
# porque el argumento del artículo es que un histórico condicionado NO es un
# modelo aprendido — y aun así gana al modelo secuencial.
axA.add_patch(FancyBboxPatch((5, 17.5), 90, 18.0,
                             boxstyle="round,pad=0.5,rounding_size=1.4",
                             linewidth=1.0, edgecolor="0.45", facecolor="none",
                             linestyle=(0, (4, 3))))
axA.text(50, 33.6, "5 · Prediction — a single evaluation frame per route",
         ha="center", va="center", fontsize=7.6, fontweight="bold", color="0.2")
axA.text(50, 31.0, "common 100 m grid · identical round trips, cut positions and folds for every model",
         ha="center", va="center", fontsize=6.3, color="0.3")

t_base = box(21, 24.0, 27.5, 11.5, C_MODEL, "Interpretable baselines",
             ["delay persistence · hour-of-day", "historical mean · linear",
              "kNN (K = 10) · kernel (b = 1)"], title_size=7.0, text_size=6.1)
t_xgb = box(50, 24.0, 27.5, 11.5, C_MODEL, "Gradient-boosted tabular",
            ["XGBoost, 4 variants:", "squared / absolute error",
             "× tier A / tier B"], title_size=7.0, text_size=6.1)
t_lstm = box(79, 24.0, 27.5, 11.5, C_MODEL, "Sequential LSTM",
             ["grid-cell tokens + attention", "full-horizon masked MSE",
              "after Sun & Kim (2021)"], title_size=7.0, text_size=6.1)

box(50, 7.0, 90, 9.0, C_EVAL, "6 · Evaluation",
    ["MAE per route · paired route-level tests · family-clustered bootstrap",
     "on-time rate within [−1, +5] min, and width-matched symmetric bands"])

for a in ((28, 89.7, 46, 85.2), (76, 89.7, 54, 85.2), (50, 76.4, 50, 72.9),
          (50, 63.6, 50, 60.1), (50, 51.4, 50, 47.7), (50, 39.8, 50, 36.0),
          (21, 17.2, 42, 12.2), (50, 17.2, 50, 12.2), (79, 17.2, 58, 12.2)):
    arrow(axA, *a)

# ══════════════════ Panel (b) — un día de un bus, cortado en vueltas ═════════
# Datos reales: `extract_figure_01b.py` corre el corte del pipeline sobre un bus-día y
# guarda la hora, el avance y el pedazo de cada registro (sin coordenadas ni bus).
axB.text(0.5, 99.0, "b", fontsize=12, fontweight="bold", va="top", ha="left")
axB.set_ylim(0, 100)
_HERE = Path(__file__).resolve().parent
_meta = json.loads((_HERE / "figure_01b_round_trips.json").read_text(encoding="utf-8"))
with open(_HERE / "figure_01b_round_trips.csv", encoding="utf-8") as _fh:
    _rec = list(csv.DictReader(_fh))
CUT, LOOP = float(_meta["cut_km"]), float(_meta["loop_km"])
OTHER = _meta.get("other_terminal_km")

# el gráfico, dentro del hueco del panel (b): margen a la izquierda para el eje, arriba
# para el título y la leyenda, y a la derecha para nombrar las líneas horizontales.
_pb = axB.get_position()
axP = fig.add_axes([_pb.x0 + 0.075, _pb.y0 + 0.052, _pb.width - 0.175, _pb.height - 0.150])
for side in ("top", "right"):
    axP.spines[side].set_visible(False)

pieces: dict[int, dict] = {}
for r in _rec:
    k = int(r["piece"])
    pc = pieces.setdefault(k, {"t": [], "y": [], "valid": r["valid"] == "True"})
    pc["t"].append(float(r["t_min"]) / 60.0)
    # km dentro de la vuelta: 0 en el punto de corte, LOOP en el siguiente cruce
    pc["y"].append((float(r["prog_km"]) - CUT) % LOOP)

C_TRIP = ("#0072B2", "#56B4E9")          # vueltas alternas (Okabe-Ito)
C_PART = "0.62"
n_valid = 0
for k in sorted(pieces):
    pc = pieces[k]
    if pc["valid"]:
        axP.plot(pc["t"], pc["y"], "-", color=C_TRIP[n_valid % 2], lw=1.25, zorder=3)
        # el cruce que abre la vuelta
        axP.plot(pc["t"][0], 0.0, marker="o", ms=3.2, color=C_VOUT, zorder=4,
                 markeredgewidth=0)
        n_valid += 1
    else:
        axP.plot(pc["t"], pc["y"], "-", color=C_PART, lw=1.1, zorder=2)

t_all = [t for pc in pieces.values() for t in pc["t"]]
x0, x1 = min(t_all) - 0.25, max(t_all) + 0.25
axP.set_xlim(x0, x1)
axP.set_ylim(-1.0, LOOP + 1.0)
axP.axhline(0.0, color=C_VOUT, lw=0.8, zorder=1)
axP.axhline(LOOP, color=C_VOUT, lw=0.8, zorder=1)
# las líneas horizontales se nombran en el margen derecho, fuera de los dientes
_lab = dict(fontsize=5.9, ha="left", va="center", clip_on=False,
            transform=axP.get_yaxis_transform())
# (10-oct) el eje está en km DENTRO DE LA VUELTA; los km de los rótulos son de la
# polilínea, y decirlo evita leer «km 22.3» sobre un eje donde el corte está en 0
axP.text(1.01, LOOP, f"cut point\n(polyline\nkm {CUT:.1f})", color=C_VOUT, **_lab)
axP.text(1.01, 0.0, f"cut point\n(polyline\nkm {CUT:.1f})", color=C_VOUT, **_lab)
if OTHER is not None:
    y_o = (float(OTHER) - CUT) % LOOP
    axP.axhline(y_o, color="0.35", lw=0.7, ls=(0, (4, 3)), zorder=1)
    axP.text(1.01, y_o, f"second\nterminal\n(polyline\nkm {float(OTHER):.1f})", color="0.3",
             **_lab)

# el salto imposible que parte el tramo: se señala desde el margen de arriba
_part = [k for k in sorted(pieces) if not pieces[k]["valid"]]
_mid = [k for k in _part if 0 < k < max(pieces)]
if len(_mid) >= 2:
    ka, kb = _mid[0], _mid[1]
    xt = (pieces[ka]["t"][-1] + pieces[kb]["t"][0]) / 2
    axP.plot([pieces[ka]["t"][-1], pieces[kb]["t"][0]],
             [pieces[ka]["y"][-1], pieces[kb]["y"][0]], ls=(0, (1, 1.5)), lw=0.8,
             color=C_PART, zorder=2)
    axP.annotate("jump", xy=(xt, (pieces[ka]["y"][-1] + pieces[kb]["y"][0]) / 2),
                 xytext=(xt + 0.45, LOOP * 0.55), fontsize=5.8, color="0.3",
                 ha="left", va="center", zorder=5,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="0.45"))

from matplotlib.lines import Line2D                                  # noqa: E402
axP.legend(handles=[
    Line2D([], [], color=C_TRIP[0], lw=1.4,
           label="round trip: from one crossing of the cut point to the next"),
    Line2D([], [], color=C_PART, lw=1.4,
           label="partial piece: before the first crossing, after the last, "
                 "or cut off by a jump"),
    Line2D([], [], color=C_VOUT, marker="o", ms=3.2, lw=0, label="crossing"),
], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
    fontsize=5.8, handlelength=1.6, columnspacing=1.2, borderaxespad=0.2)

hours = range(int(x0) + 1, int(x1) + 1)
axP.set_xticks(list(hours))
axP.set_xticklabels([f"{h}:00" for h in hours], fontsize=6.0)
axP.tick_params(axis="y", labelsize=6.0)
axP.set_xlabel("time of day", fontsize=6.6)
axP.set_ylabel("km into the round trip", fontsize=6.6)
axB.text(52, 97.5,
         f"One vehicle on route {_meta['route']}, {_meta['weekday']} {_meta['day']}: "
         f"{n_valid} round trips\n"
         f"cut where the vehicle's continuous progress crosses the route's cut point",
         fontsize=6.9, fontweight="bold", ha="center", va="top", color="0.15",
         linespacing=1.35)

fit_texts()
# Los tres títulos de la etapa 5 al MISMO cuerpo: el ajuste automático encoge
# solo el que se sale, y tres tamaños distintos en una fila se leen como error.
_same = min(t.get_fontsize() for t in (t_base, t_xgb, t_lstm))
for t in (t_base, t_xgb, t_lstm):
    t.set_fontsize(_same)

out = Path(__file__).parent
for ext in ("png", "pdf"):
    fig.savefig(out / f"figure_01_merged.{ext}", dpi=300, facecolor="white")

# ⚠ Guard de tamaño: si la figura se sale del máximo de la revista hay que
# enterarse aquí y no en la maquetación.
w_mm, h_mm = FIG_W * 25.4, FIG_H * 25.4
if w_mm > 174.5 or h_mm > 234.5:
    raise SystemExit(f"✗ {w_mm:.0f} × {h_mm:.0f} mm excede el máximo de "
                     f"Public Transport (174 × 234 mm)")
print(f"saved: {out / 'figure_01_merged.png'}  ·  {w_mm:.0f} × {h_mm:.0f} mm (máx 174 × 234)")
