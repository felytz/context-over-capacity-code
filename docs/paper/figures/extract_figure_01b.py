"""Los datos del panel (b) de la Figura 1: un bus, un día, cortado por cruce de km.

El panel (b) de agosto ilustraba el detector de reversión con un dibujo. Con la vuelta
el corte es otro, y se enseña con datos reales: el avance continuo M(t) de un bus a lo
largo de un día y los cruces del punto de corte, que parten el día en vueltas (plan,
fase 1b).

Cómo se saca
------------
Se corre `_lib_new_trips._build_trips_cruce` —el corte del pipeline, con los
interruptores de `config.env` §7— sobre los registros limpios (`df_rec_clean`) de UN
bus-día. El corte es por bus (la proyección coherente, los saltos y los cruces se calculan
dentro de los tramos de cada bus), así que sale lo mismo que en la corrida de la ruta.
La entrada es la de una corrida local de la fase de datos, p. ej. la del piloto
(1601, marzo de 2024):

    data/02_interim/routes/1601/dataframes/df_rec_clean_20240301-20240401_tmin.parquet

`data/` está gitignorado: este script sólo corre donde esté esa corrida, igual que
`network_funnel.py` con el crudo. Lo que se versiona es su salida, que no lleva ni
coordenadas ni el código del bus: la hora del día, el avance y a qué pedazo pertenece
cada registro.

Salida:
  docs/paper/figures/figure_01b_round_trips.csv
  docs/paper/figures/figure_01b_round_trips.json   (la ruta, el día, el corte y el lazo)

Run:  python docs/paper/figures/extract_figure_01b.py [--route 1601 --bus SIT-0145 --day 2024-03-24]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "figure_01b_round_trips"

# Los interruptores de la vuelta, como en config.env §7 (antes de importar el módulo,
# que los lee al importarse).
os.environ.setdefault("TRIP_CUT", "cruce")
os.environ.setdefault("TERMINAL_RULE", "file")
os.environ.setdefault("TERMINALS_FILE", str(ROOT / "cortes" / "cortes_vuelta.json"))
os.environ.setdefault("COHERENT_PROJECTION", "1")
os.environ.setdefault("CRUCE_DIRECCIONAL", "1")
sys.path.insert(0, str(ROOT / "func"))
sys.path.insert(0, str(ROOT / "func" / "lib"))

import polars as pl                                                # noqa: E402

import _lib_new_trips as nt                                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--route", default="1601")
    ap.add_argument("--bus", default="SIT-0145")
    ap.add_argument("--day", default="2024-03-24")
    ap.add_argument("--clean", default=None,
                    help="df_rec_clean de la ruta (default: el de marzo del piloto local)")
    a = ap.parse_args()

    clean = Path(a.clean) if a.clean else (
        ROOT / "data" / "02_interim" / "routes" / a.route / "dataframes"
        / "df_rec_clean_20240301-20240401_tmin.parquet")
    if not clean.is_file():
        raise SystemExit(f"✗ falta {clean}\n  Corre antes la fase de datos de la ruta "
                         f"(run_new_pipeline.py) o pasa --clean.")
    day = date.fromisoformat(a.day)
    df = (pl.read_parquet(clean)
            .filter((pl.col("bus_code") == a.bus) & (pl.col("time").dt.date() == day)))
    if df.is_empty():
        raise SystemExit(f"✗ {a.bus} no tiene registros el {a.day} en {clean.name}")

    poly, cum, L = nt.load_route_polyline(str(ROOT / "data" / "01_raw" / "polylines.geojson"),
                                          int(a.route))
    tc = nt.load_terminal_coords(int(a.route))
    dt, _meta, stats, _otab, frag = nt._build_trips_cruce(df, poly, cum, L, tc)

    dt = dt.sort("time")
    pieces = {t: i for i, t in enumerate(dict.fromkeys(dt["trip_id"].to_list()))}
    # Minutos desde la medianoche. ⚠ No `dt.hour() * 60`: `hour` es Int8 y se desborda.
    t0 = ((dt["time"] - dt["time"].dt.truncate("1d")).dt.total_seconds() / 60.0)
    rows = [{"t_min": round(float(t), 3), "prog_km": round(float(p) / 1000, 4),
             "piece": pieces[tid], "complete": cf is not None, "valid": bool(v),
             "shortcut": bool(at)}
            for t, p, tid, cf, v, at in zip(t0, dt["prog_m"], dt["trip_id"],
                                            dt["cut_from"], dt["trip_valid"], dt["atajo"])]
    with open(OUT.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    cr = stats["cruce"]
    meta = {"route": a.route, "day": a.day, "weekday": day.strftime("%A"),
            "cut_km": cr["cortes_km"][0], "loop_km": cr["lazo_km"],
            "stretches": cr["tramos"], "split_by": cr["partidos_por"],
            "round_trips_valid": cr["viajes_validos"],
            "pieces_partial": cr["pedazos_parciales"],
            "dur_med_min": cr["dur_med_min"],
            "source": f"{clean.relative_to(ROOT).as_posix()} (bus and day given above "
                      f"to the script; the output carries neither coordinates nor bus)"}
    # La otra terminal, si la ruta es de dos (cortes_vuelta.csv, «cortando en km X»).
    import re
    with open(ROOT / "cortes" / "cortes_vuelta.csv", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["ruta"] == a.route and r["tipo"] == "dos":
                m = re.search(r"cortando en km ([0-9.]+)", r["fuente"])
                if m:
                    meta["other_terminal_km"] = float(m.group(1))
    meta.pop("source")      # la ruta del archivo local no viaja; el docstring dice de dónde
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=1))
    print(f"✓ {OUT.with_suffix('.csv')} ({len(rows)} registros, {len(pieces)} pedazos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
