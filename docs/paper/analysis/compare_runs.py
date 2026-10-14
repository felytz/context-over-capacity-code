"""Compara DOS corridas del mismo régimen, leg a leg y algoritmo a algoritmo.

    python docs/paper/analysis/compare_runs.py --a test5/extracted --b runs/nueva
    python docs/paper/analysis/compare_runs.py --a A --b B --panel B   # solo las 42
    python docs/paper/analysis/compare_runs.py --a A --b B --csv dif.csv

Cada `--root` es la carpeta donde se descomprimió un tarball, con el árbol
`data/02_interim/routes/<ruta>/{xgb,lstm_frame}/<rango>/…` dentro.

⚠ LA REGLA QUE HACE VÁLIDA LA COMPARACIÓN
------------------------------------------
**Solo se comparan las (ruta, leg, algoritmo) presentes en LAS DOS corridas.**
Comparar medianas sobre conjuntos de legs distintos no dice nada: basta con que
a una corrida le falte una leg difícil para que su mediana «mejore». Si una
corrida tiene legs que la otra no —lo normal, p. ej. si una saltó el estrato
N=1— aquí se listan aparte y se excluyen del agregado.

Por la misma razón `--panel` acepta un `leg_inclusion.csv` y restringe al panel
pre-registrado (direccional · M ≥ 100 · M ≥ n_folds), que es el de las tablas
del paper. Sin él se compara sobre todas las legs comunes, que es un conjunto
MÁS GRANDE y cuyas medianas NO son las publicadas.

⚠ El LSTM no es bit-reproducible: el no-determinismo es de la GPU, no del
frame. Diferencias de hasta ±4 s en su MAE son ruido conocido, no un fallo.
Lo que sí debe salir idéntico es todo lo demás — baselines y XGBoost son
deterministas dados el mismo frame y las mismas semillas.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import polars as pl

RANGE = "2024_01-08_tmin"
TAG = "q05c100"

# Columnas que se comparan, con su formato y unidad.
METRICS = [("MAE_s", "s", 2), ("bias_s", "s", 1), ("pct_ontime_dev", "pp", 2)]

Key = Tuple[str, str, str]          # (ruta, leg, algoritmo)


def read_run(root: Path, tag: str, rng: str) -> pl.DataFrame:
    """Todos los `_overall.parquet` de las dos familias, en una tabla."""
    keep = ["route", "leg", "algorithm", "n_pairs", "MAE_s", "bias_s",
            "pct_ontime_dev"]
    base = root / "data" / "02_interim" / "routes"
    if not base.is_dir():
        sys.exit(f"✗ no veo {base}\n"
                 f"  ¿Descomprimiste el tarball ahí? Debe contener "
                 f"data/02_interim/routes/…")
    parts = []
    for rdir in sorted(base.iterdir()):
        if not rdir.is_dir():
            continue
        # xgb escribe `_exo_overall` cuando corrió con --exo, `_overall` si no;
        # los dos globs son disjuntos, así que se unen sin duplicar.
        paths: Set[Path] = set()
        for d, pat in ((rdir / "xgb" / rng, f"*_{tag}_exo_overall.parquet"),
                       (rdir / "xgb" / rng, f"*_{tag}_overall.parquet"),
                       (rdir / "lstm_frame" / rng, f"*_{tag}_overall.parquet")):
            if d.is_dir():
                paths |= set(d.glob(pat))
        for p in sorted(paths):
            try:
                x = pl.read_parquet(p)
            except Exception as e:                      # parquet truncado
                print(f"  ⚠ ilegible, se salta: {p.name} ({e})")
                continue
            cols = [c for c in keep if c in x.columns]
            parts.append(x.select(cols).with_columns(
                pl.lit(rdir.name).alias("route"),
                pl.col("leg").cast(pl.Utf8)))
    if not parts:
        sys.exit(f"✗ ningún agregado del régimen {tag} bajo {base}")
    df = pl.concat(parts, how="diagonal_relaxed")
    # Una (ruta, leg, algoritmo) repetida sería un artefacto de dos regímenes
    # mezclados en el mismo árbol: quedarse con una callaría el problema.
    dup = df.group_by(["route", "leg", "algorithm"]).len().filter(pl.col("len") > 1)
    if dup.height:
        print(f"  ⚠ {dup.height} (ruta, leg, algoritmo) DUPLICADAS en {root} — "
              f"¿dos regímenes en el mismo árbol?")
    return df


def load_panel(csv_path: Path) -> Set[Tuple[str, str]]:
    """(ruta, leg) marcadas `included` en un leg_inclusion.csv."""
    keep = set()
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("included") or "").strip().lower() == "true":
                keep.add((str(row["route"]), str(row["leg"])))
    if not keep:
        sys.exit(f"✗ {csv_path} no marca ninguna leg como included")
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="raíz de la corrida A (referencia)")
    ap.add_argument("--b", required=True, help="raíz de la corrida B (nueva)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--range", dest="rng", default=RANGE)
    ap.add_argument("--panel", default=None,
                    help="leg_inclusion.csv para restringir al panel "
                         "pre-registrado (el de las tablas del paper)")
    ap.add_argument("--top", type=int, default=10,
                    help="cuántas legs con mayor movimiento listar")
    ap.add_argument("--csv", default=None, help="vuelca el detalle por leg")
    a = ap.parse_args()

    A, B = Path(a.a).resolve(), Path(a.b).resolve()
    print("=" * 78)
    print(f"  COMPARACIÓN DE CORRIDAS · {a.tag} · {a.rng}")
    print(f"    {a.label_a:8} {A}")
    print(f"    {a.label_b:8} {B}")
    print("=" * 78)

    da, db = read_run(A, a.tag, a.rng), read_run(B, a.tag, a.rng)

    # ── 1 · cobertura ────────────────────────────────────────────────────────
    legs_a = set(map(tuple, da.select(["route", "leg"]).unique().rows()))
    legs_b = set(map(tuple, db.select(["route", "leg"]).unique().rows()))
    common = legs_a & legs_b
    only_a, only_b = legs_a - legs_b, legs_b - legs_a

    print(f"\n1 · COBERTURA")
    print(f"  legs en {a.label_a:8}: {len(legs_a)}")
    print(f"  legs en {a.label_b:8}: {len(legs_b)}")
    print(f"  COMUNES            : {len(common)}   ← sobre estas se compara")
    for label, s in ((a.label_a, only_a), (a.label_b, only_b)):
        if s:
            muestra = ", ".join(f"{r}/{l}" for r, l in sorted(s)[:8])
            print(f"  solo en {label:8}   : {len(s)}  ({muestra}"
                  f"{' …' if len(s) > 8 else ''})")
    if not common:
        print("\n✗ sin legs comunes: no hay nada que comparar.")
        return 1

    if a.panel:
        panel = load_panel(Path(a.panel))
        antes = len(common)
        common &= panel
        print(f"  panel pre-registrado: {antes} → {len(common)} legs")
        if not common:
            print("\n✗ el panel no deja ninguna leg común.")
            return 1

    keep = pl.DataFrame({"route": [r for r, _ in sorted(common)],
                         "leg":   [l for _, l in sorted(common)]})
    da, db = da.join(keep, on=["route", "leg"]), db.join(keep, on=["route", "leg"])

    # ── 2 · por algoritmo, sobre las MISMAS legs ─────────────────────────────
    j = da.join(db, on=["route", "leg", "algorithm"], how="inner", suffix="_b")
    if j.is_empty():
        print("\n✗ ninguna (leg, algoritmo) coincide entre las dos corridas.")
        return 1

    print(f"\n2 · POR ALGORITMO  (mediana sobre las legs comunes)")
    hdr = f"  {'algoritmo':12}"
    for m, u, _ in METRICS:
        hdr += f" {a.label_a[:7] + ' ' + m.split('_')[0]:>16} {'Δ ' + u:>9}"
    print(hdr + f" {'legs':>5}")
    print("  " + "-" * (len(hdr) + 4))

    rows = []
    for algo in sorted(j["algorithm"].unique().to_list()):
        sub = j.filter(pl.col("algorithm") == algo)
        line = f"  {algo:12}"
        rec = {"algorithm": algo, "legs": sub.height}
        for m, u, dec in METRICS:
            if m not in sub.columns or f"{m}_b" not in sub.columns:
                line += f" {'—':>16} {'—':>9}"
                continue
            va, vb = sub[m].median(), sub[f"{m}_b"].median()
            if va is None or vb is None:
                line += f" {'—':>16} {'—':>9}"
                continue
            d = vb - va
            line += f" {va:>16.{dec}f} {d:>+9.{dec}f}"
            rec[f"{m}_a"], rec[f"{m}_b"], rec[f"{m}_d"] = va, vb, d
        print(line + f" {sub.height:>5}")
        rows.append(rec)

    # ── 3 · dónde se movió más ──────────────────────────────────────────────
    if "MAE_s" in j.columns and "MAE_s_b" in j.columns:
        mov = (j.with_columns((pl.col("MAE_s_b") - pl.col("MAE_s")).alias("d_MAE_s"))
                .filter(pl.col("d_MAE_s").is_not_null())
                .sort(pl.col("d_MAE_s").abs(), descending=True))
        print(f"\n3 · LAS {a.top} LEGS QUE MÁS SE MOVIERON (MAE)")
        print(f"  {'ruta':>6} {'leg':12} {'algoritmo':12} "
              f"{a.label_a[:8]:>9} {a.label_b[:8]:>9} {'Δ s':>8}")
        for r in mov.head(a.top).iter_rows(named=True):
            print(f"  {r['route']:>6} {r['leg']:12} {r['algorithm']:12} "
                  f"{r['MAE_s']:>9.2f} {r['MAE_s_b']:>9.2f} {r['d_MAE_s']:>+8.2f}")

        # ── Lo determinista debe salir IDÉNTICO; el LSTM no ─────────────────
        # Se cuenta cuántos pares son BIT-IDÉNTICOS, no solo el peor |Δ|: un
        # máximo aislado justo por debajo de cualquier umbral no distingue
        # "reprodujo salvo tres legs" de "se movió medio conjunto", y son
        # diagnósticos opuestos.
        det = mov.filter(~pl.col("algorithm").str.starts_with("lstm"))
        n_det = det.height
        n_id = det.filter(pl.col("d_MAE_s") == 0).height if n_det else 0
        peor = det["d_MAE_s"].abs().max() if n_det else 0.0
        print(f"\n4 · REPRODUCIBILIDAD DE LA PARTE DETERMINISTA")
        if n_det:
            print(f"  (baselines y XGBoost: mismo frame + mismas semillas ⇒ mismo número)")
            print(f"  pares (leg, algoritmo)  : {n_det}")
            print(f"  BIT-IDÉNTICOS           : {n_id}  ({100.0 * n_id / n_det:.1f}%)")
            print(f"  se mueven               : {n_det - n_id}   ·   máx |Δ| {peor:.3f} s")
        if not n_det:
            # Un régimen solo-LSTM (p. ej. `_es15`): no hay nada determinista
            # que exigir, y el veredicto de abajo daría una falsa alarma.
            print("  (este régimen solo tiene LSTM, que no es bit-reproducible en")
            print("   GPU: hasta ±4 s de MAE por leg es ruido conocido)")
        elif n_id == n_det:
            print("  ✓ reproducción EXACTA. Todo el movimiento está en el LSTM,")
            print("    que no es bit-reproducible (el no-determinismo es de la GPU).")
        elif (n_det - n_id) <= max(3, n_det // 100) and (peor or 0) < 2.0:
            print("  ✓ reproduce salvo unos pocos pares, y por debajo del segundo.")
            print("    Causa típica: otra versión de xgboost/OpenMP cambia el orden")
            print("    de reducción en punto flotante. Comprueba `n_pairs`: si es")
            print("    idéntico, la MUESTRA es la misma y solo difiere el ajuste.")
            for r in det.filter(pl.col("d_MAE_s") != 0).head(8).iter_rows(named=True):
                print(f"      {r['route']:>6} {r['leg']:10} {r['algorithm']:12} "
                      f"Δ{r['d_MAE_s']:+7.3f} s")
        else:
            print("  ⚠ demasiados pares deterministas se mueven. Eso NO es jitter")
            print("    de GPU: sospecha de régimen, de datos o de frame distintos.")
            print("    Empieza comparando `n_pairs` — si difiere, la muestra no es")
            print("    la misma y ninguna comparación de medianas significa nada.")

    if a.csv:
        out = Path(a.csv)
        j.write_csv(out)
        print(f"\n  ✓ detalle por leg → {out}")

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
