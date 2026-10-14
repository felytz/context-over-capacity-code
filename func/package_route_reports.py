#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
package_route_reports.py — arma, en el server, TODO lo necesario para el
reporte de resultados de tesis, POR RUTA, sin transferir los 200+ GB crudos.

Problema: los artefactos de resultados (LSTM / arrival) son `df_errors.parquet`
de 20-25 GB por ruta (una fila por predicción). Para el reporte no hacen falta
esas filas, sino sus MÉTRICAS AGREGADAS. Este script:

  1. CALCULA un digest compacto por ruta leyendo (en streaming) los parquets
     pesados: métricas de LSTM (vs baselines) y de arrival — MAE/RMSE/bias/APTA
     on-time y curvas por horizonte —, resumen de trips y estructura de datos.
  2. EMPAQUETA por ruta un `<ruta>.zip` con ese digest + los artefactos que ya
     son chicos (funnel/perf/stats/trip_meta, df_segments,
     trayectorias png/html, PDFs de reporte, metadatos de LSTM…).
  3. Emite un `analysis_digest.json` global + `digest_bundle.zip` (solo los
     digests, muy liviano) — eso es lo que le pasas a Claude aunque los zips
     por ruta pesen por los PDFs.

Los parquets pesados (`df_errors*`, `cv_ckpt_fold*_errors`,
`lstm_tf_vs_baselines*`) NUNCA se copian a los zips: son la FUENTE del digest.

Requiere: polars y pyarrow (ya están en el entorno del clúster). Solo stdlib
para el resto.

Uso
----
    python package_route_reports.py                 # todas las rutas, modo both
    python package_route_reports.py --routes 101,102,1701
    python package_route_reports.py --mode digest   # solo el digest liviano
    python package_route_reports.py --mode zip       # solo los zips por ruta
    python package_route_reports.py --include-df-final --include-arrival-pdf
    python package_route_reports.py --root /lustre/home/usuario/db/data -o /scratch/out

Flags
-----
    --root DIR           raíz de datos (default: autodetecta <..>/data con 01_raw+02_interim)
    --routes LIST        "all" | "101,102,1701"
    --variants LIST      variantes a agregar (default: tmin,t1,t2)
    --mode {both,digest,zip}
    -o, --out DIR        carpeta de salida (default: ./route_reports)
    --max-file-mb N      no copiar al zip archivos > N MB (default 60)
    --include-df-final   incluye df_final_new_trips_*.parquet (7-35 MB/ruta)
    --include-arrival-pdf incluye los PDF de arrival (hasta ~45 MB)
    --include-maps       incluye maps/ y dashboards/ (100+ MB/ruta) — pesado
    --quiet              menos verboso
"""
from __future__ import annotations
import argparse
import io
import json
import math
import os
import re
import socket
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import polars as pl

# ── constantes de dominio ──────────────────────────────────────────
APTA_LO, APTA_HI = -60.0, 300.0     # dev = t_true - t_pred on-time si ∈ [-60,+300] s
ERR_COLS = ("t_true", "t_pred", "error_abs", "error_sq", "horizon_min")

# artefactos pesados que son FUENTE del digest y jamás se copian al zip
HEAVY_RE = re.compile(
    r"(df_errors.*\.parquet$|cv_ckpt_fold\d+_errors\.parquet$|"
    r"lstm_tf_vs_baselines.*\.parquet$)")

# extensiones "chicas" que sí se copian (si <= max-file-mb)
COPY_EXT = {".json", ".png", ".html", ".pdf", ".csv", ".txt", ".md"}
# parquets chicos permitidos por nombre
COPY_PARQUET = ("df_segments.parquet",)


def log(*a):
    print(*a, flush=True)


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:,.1f} {u}"
        n /= 1024


def r3(x):
    """Redondeo seguro para JSON (evita NaN/inf que rompen json estándar)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 3)


def find_root(start: Path) -> Path | None:
    p = start.resolve()
    while p != p.parent:
        if (p / "01_raw").is_dir() and (p / "02_interim").is_dir():
            return p
        if (p / "data" / "01_raw").is_dir():
            return p / "data"
        p = p.parent
    return None


# ── agregación de errores ──────────────────────────────────────────

def _agg_exprs(with_quantiles: bool = True):
    """Agregaciones de error. Las 'baratas' (n/mae/rmse/bias/apta) son de UN
    solo paso y memoria O(1) en streaming. median()/quantile() son HOLÍSTICAS:
    materializan cada grupo entero en RAM → catastróficas sobre parquets de
    decenas de GB. Por eso los cuantiles son opcionales (se apagan por tamaño)."""
    dev = pl.col("t_true") - pl.col("t_pred")
    exprs = [
        pl.len().alias("n"),
        pl.col("error_abs").mean().alias("mae_s"),
        pl.col("error_sq").mean().sqrt().alias("rmse_s"),
        dev.mean().alias("bias_s"),
        (((dev >= APTA_LO) & (dev <= APTA_HI)).mean()).alias("apta_ontime_frac"),
    ]
    if with_quantiles:
        exprs += [
            pl.col("error_abs").median().alias("p50_abs_s"),
            pl.col("error_abs").quantile(0.90).alias("p90_abs_s"),
        ]
    return exprs


def _row_to_metrics(d: dict) -> dict:
    m = {
        "n": int(d["n"]) if d.get("n") is not None else 0,
        "mae_s": r3(d.get("mae_s")), "mae_min": r3((d.get("mae_s") or 0) / 60),
        "rmse_s": r3(d.get("rmse_s")), "rmse_min": r3((d.get("rmse_s") or 0) / 60),
        "bias_s": r3(d.get("bias_s")),
        "p50_abs_s": r3(d.get("p50_abs_s")), "p90_abs_s": r3(d.get("p90_abs_s")),
        "apta_ontime_pct": r3((d.get("apta_ontime_frac") or 0) * 100),
    }
    return m


def _file_gb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 ** 3)
    except OSError:
        return 0.0


def aggregate_errors(path: Path, by_algo: bool, quantile_max_gb: float = 2.0) -> dict:
    """Devuelve {overall, by_algorithm?, by_horizon(_algo)} desde un df_errors/
    vs_baselines parquet, en streaming (no carga todo a memoria). Los cuantiles
    (p50/p90) se OMITEN si el archivo pesa más que `quantile_max_gb` (holísticas,
    revientan RAM en parquets grandes)."""
    lf = pl.scan_parquet(path)
    cols = lf.collect_schema().names()
    if not all(c in cols for c in ERR_COLS):
        return {"error": f"faltan columnas {ERR_COLS} en {path.name}: {cols}"}
    has_algo = by_algo and "algorithm" in cols
    wq = _file_gb(path) <= quantile_max_gb
    out: dict = {"quantiles": wq}

    out["overall"] = _row_to_metrics(
        lf.select(_agg_exprs(wq)).collect(engine="streaming").to_dicts()[0])

    if has_algo:
        df = lf.group_by("algorithm").agg(_agg_exprs(wq)).sort("algorithm") \
               .collect(engine="streaming")
        out["by_algorithm"] = {r["algorithm"]: _row_to_metrics(r)
                               for r in df.to_dicts()}

    # por horizonte: siempre sin cuantiles (muchos grupos × holística = caro)
    lf_h = lf.with_columns(
        pl.col("horizon_min").floor().cast(pl.Int32).alias("h_bucket"))
    gb = ["algorithm", "h_bucket"] if has_algo else ["h_bucket"]
    dh = lf_h.group_by(gb).agg(_agg_exprs(False)).sort(gb).collect(engine="streaming")
    horiz: dict = {}
    for r in dh.to_dicts():
        key = r["algorithm"] if has_algo else "all"
        horiz.setdefault(key, []).append(
            {"horizon_min": int(r["h_bucket"]), "n": int(r["n"]),
             "mae_s": r3(r["mae_s"]), "apta_ontime_pct": r3((r["apta_ontime_frac"] or 0) * 100)})
    out["by_horizon"] = horiz
    return out


# ── localizadores ──────────────────────────────────────────────────

def raw_records(root: Path, route: str) -> Path | None:
    d = root / "01_raw" / "records" / route
    if not d.is_dir():
        return None
    longs = sorted(d.glob(f"df_rec_raw_{route}_*2024_01-12.parquet"))
    if longs:
        return longs[0]
    any_pq = sorted(d.glob("*.parquet"))
    return any_pq[0] if any_pq else None


def find_variant_dirs(cat_dir: Path, variants: list[str]) -> list[Path]:
    if not cat_dir.is_dir():
        return []
    out = []
    for v in variants:
        out += sorted(cat_dir.glob(f"*_{v}"))
    return out


# ── secciones del digest ───────────────────────────────────────────

def build_data_structure(root: Path, route: str) -> dict:
    # Solo esquema y conteos. Antes se volcaban filas de muestra del crudo con
    # TODAS sus columnas —plate_number incluida— y el digest viajaba en el .tgz.
    # Los buses se cuentan por `code` (= bus_code), no por `id`.
    ds: dict = {"route": route}
    pq = raw_records(root, route)
    if pq is None:
        ds["records"] = "no hay parquet crudo"
        return ds
    lf = pl.scan_parquet(pq)
    schema = lf.collect_schema()
    ds["records"] = {
        "file": pq.name,
        "columns": {n: str(t) for n, t in zip(schema.names(), schema.dtypes())},
    }
    try:
        agg = lf.select(
            pl.len().alias("rows"),
            pl.col("utc_datetime").min().alias("mn"),
            pl.col("utc_datetime").max().alias("mx"),
            pl.col("code").n_unique().alias("buses"),
            pl.col("group_id").n_unique().alias("groups"),
        ).collect(engine="streaming").to_dicts()[0]
        ds["records"].update(rows=int(agg["rows"]),
                             date_min=str(agg["mn"])[:19], date_max=str(agg["mx"])[:19],
                             n_buses=int(agg["buses"]), n_groups=int(agg["groups"]))
    except Exception as e:
        ds["records"]["stats_error"] = f"{type(e).__name__}: {e}"

    # geometría/paradas de la ruta (desde los json compartidos)
    raw = root / "01_raw"
    try:
        sr = json.load(open(raw / "stops_routes.json", encoding="utf-8"))
        seg = json.load(open(raw / "segments_route.json", encoding="utf-8"))
        ds["geometry"] = {
            "n_stops": len(sr.get(route, [])),
            "n_segments_ref": len(seg.get(route, [])),
            "in_stops_routes": route in sr, "in_segments_route": route in seg,
        }
    except Exception as e:
        ds["geometry"] = {"error": f"{type(e).__name__}: {e}"}
    return ds


def build_trips_summary(rdir: Path, variants: list[str]) -> dict:
    ts: dict = {}
    # funnel + perf (copiamos los números clave del último variante disponible)
    perf_dir = rdir / "perf"
    for tag, pat in (("funnel", "funnel_*.json"), ("perf", "perf_*.json")):
        hits = sorted(perf_dir.glob(pat)) if perf_dir.is_dir() else []
        # prioriza tmin
        hits.sort(key=lambda p: (0 if "tmin" in p.name else 1, p.name))
        if hits:
            try:
                ts[tag] = json.load(open(hits[0], encoding="utf-8"))
            except Exception as e:
                ts[tag] = {"error": f"{type(e).__name__}: {e}"}

    # df_final_new_trips → resumen de trips (duración, distancia, dirección, legs)
    df_dir = rdir / "dataframes"
    finals = sorted(df_dir.glob("df_final_new_trips_*.parquet")) if df_dir.is_dir() else []
    finals.sort(key=lambda p: (0 if "tmin" in p.name else 1, p.name))
    if finals:
        f = finals[0]
        try:
            lf = pl.scan_parquet(f)
            cols = lf.collect_schema().names()
            per_trip = lf.group_by("trip_id").agg(
                (pl.col("time").max() - pl.col("time").min()).dt.total_seconds().alias("dur_s"),
                (pl.col("cum_dist_m").max() if "cum_dist_m" in cols else pl.lit(None)).alias("dist_m"),
                pl.len().alias("pings"),
                (pl.col("leg").first() if "leg" in cols else pl.lit(None)).alias("leg"),
                (pl.col("direction").first() if "direction" in cols else pl.lit(None)).alias("direction"),
            ).collect(engine="streaming")
            dur = per_trip["dur_s"].drop_nulls()
            ts["trips"] = {
                "source": f.name,
                "n_trips": int(per_trip.height),
                "pings_total": int(per_trip["pings"].sum()),
                "dur_s": {"mean": r3(dur.mean()), "p50": r3(dur.median()),
                          "p10": r3(dur.quantile(0.1)), "p90": r3(dur.quantile(0.9))},
                "dur_min": {"mean": r3((dur.mean() or 0) / 60), "p50": r3((dur.median() or 0) / 60)},
                "pings_per_trip": {"mean": r3(per_trip["pings"].mean()),
                                   "p50": r3(per_trip["pings"].median())},
            }
            if "dist_m" in per_trip.columns and per_trip["dist_m"].drop_nulls().len():
                dd = per_trip["dist_m"].drop_nulls()
                ts["trips"]["dist_km"] = {"mean": r3((dd.mean() or 0) / 1000),
                                          "p50": r3((dd.median() or 0) / 1000)}
            if "leg" in per_trip.columns:
                ts["trips"]["trips_per_leg"] = {
                    str(k): int(v) for k, v in
                    per_trip.group_by("leg").len().sort("leg").iter_rows()}
            if "direction" in per_trip.columns:
                ts["trips"]["trips_per_direction"] = {
                    str(k): int(v) for k, v in
                    per_trip.group_by("direction").len().sort("direction").iter_rows()}
        except Exception as e:
            ts["trips"] = {"error": f"{type(e).__name__}: {e}",
                           "trace": traceback.format_exc()[-500:]}
    return ts


def _agg_logged(path: Path, by_algo: bool, args, label: str) -> dict:
    """Agrega un parquet de errores con log de tamaño+tiempo y guarda de tamaño.
    Si el archivo supera `--max-agg-gb` (y no es 0), se SALTA (solo registra
    tamaño) para no arriesgar un cuelgue."""
    gb = _file_gb(path)
    if args.max_agg_gb and gb > args.max_agg_gb:
        log(f"      · {label} {gb:.1f} GB … SALTADO (> --max-agg-gb {args.max_agg_gb})")
        return {"skipped_too_big_gb": r3(gb)}
    if not args.quiet:
        log(f"      · {label} {gb:.1f} GB …")
    t0 = time.time()
    try:
        out = aggregate_errors(path, by_algo=by_algo, quantile_max_gb=args.quantile_max_gb)
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {e}"}
    out["_gb"] = r3(gb)
    out["_secs"] = r3(time.time() - t0)
    if not args.quiet:
        log(f"        ✓ {label} en {out['_secs']}s"
            + ("" if out.get("quantiles", True) else "  (sin cuantiles: archivo grande)"))
    return out


def build_model_metrics(rdir: Path, variants: list[str], args) -> dict:
    res: dict = {"lstm": {}, "arrival": {}}

    # LSTM: vs_baselines por variante (raíz de lstm_tf) → por algoritmo.
    # Esto YA compara LSTM contra baselines y da curvas por horizonte, así que
    # es la fuente principal de los resultados de LSTM.
    lstm = rdir / "lstm_tf"
    if lstm.is_dir():
        for v in variants:
            for vb in sorted(lstm.glob(f"lstm_tf_vs_baselines_*_{v}.parquet")):
                res["lstm"].setdefault(v, {})["vs_baselines"] = _agg_logged(
                    vb, True, args, f"LSTM vs_baselines [{v}]")
        # Detalle LSTM por leg (df_errors): son los parquets de 20-25 GB. Es
        # REDUNDANTE con vs_baselines (que ya trae el algoritmo lstm), así que
        # está APAGADO por default. Actívalo con --lstm-by-leg si de veras lo
        # quieres (y ajusta --max-agg-gb / --quantile-max-gb).
        if args.lstm_by_leg:
            for vdir in find_variant_dirs(lstm, variants):
                v = vdir.name.rsplit("_", 1)[-1]
                for legdir in sorted(p for p in vdir.iterdir()
                                     if p.is_dir() and p.name.startswith("legT")):
                    de = legdir / "df_errors.parquet"
                    if de.is_file():
                        res["lstm"].setdefault(v, {}).setdefault("by_leg", {})[legdir.name] = \
                            _agg_logged(de, False, args, f"LSTM df_errors [{v}/{legdir.name}]")

    # ARRIVAL: df_errors por variante × leg
    arr = rdir / "arrival"
    if arr.is_dir():
        for vdir in find_variant_dirs(arr, variants):
            v = vdir.name.rsplit("_", 1)[-1]
            for legdir in sorted(p for p in vdir.iterdir()
                                 if p.is_dir() and p.name.startswith("legT")):
                de = legdir / "df_errors.parquet"
                if de.is_file():
                    res["arrival"].setdefault(v, {})[legdir.name] = _agg_logged(
                        de, True, args, f"arrival df_errors [{v}/{legdir.name}]")
    return res


def rollup_headline(route: str, models: dict, trips: dict, ds: dict) -> dict:
    """Números de portada por ruta, para la tabla-resumen global."""
    hl: dict = {"route": route}
    rec = ds.get("records", {})
    if isinstance(rec, dict):
        hl["records_rows"] = rec.get("rows")
        hl["n_buses"] = rec.get("n_buses")
        hl["date_range"] = f"{rec.get('date_min','?')[:10]}..{rec.get('date_max','?')[:10]}"
    tr = trips.get("trips", {})
    hl["n_trips"] = tr.get("n_trips") if isinstance(tr, dict) else None
    fn = trips.get("funnel", {})
    if isinstance(fn, dict):
        hl["trips_valid"] = fn.get("trips_valid")
        hl["n_legs"] = fn.get("n_legs")
        hl["pct_pings_kept"] = fn.get("pct_pings_kept")
    # mejor variante LSTM: MAE del algoritmo 'lstm' si existe
    for v, blk in (models.get("lstm") or {}).items():
        vb = (blk or {}).get("vs_baselines", {})
        by = vb.get("by_algorithm", {}) if isinstance(vb, dict) else {}
        for algo, m in by.items():
            if "lstm" in algo.lower():
                hl.setdefault("lstm_mae_min", {})[v] = m.get("mae_min")
                hl.setdefault("lstm_apta_pct", {})[v] = m.get("apta_ontime_pct")
    # arrival headline: overall del primer leg por variante
    for v, legs in (models.get("arrival") or {}).items():
        for leg, blk in (legs or {}).items():
            ov = blk.get("overall", {}) if isinstance(blk, dict) else {}
            if ov:
                hl.setdefault("arrival_mae_min", {})[f"{v}/{leg}"] = ov.get("mae_min")
                break
    return hl


# ── empaquetado ────────────────────────────────────────────────────

def collect_copy_files(rdir: Path, args) -> list[Path]:
    """Archivos ya-chicos a copiar tal cual al zip de la ruta."""
    if not rdir.is_dir():
        return []
    picked: list[Path] = []
    cap = args.max_file_mb * 1024 * 1024
    for f in rdir.rglob("*"):
        if not f.is_file():
            continue
        name, ext = f.name, f.suffix.lower()
        rel = str(f.relative_to(rdir)).replace("\\", "/")
        if HEAVY_RE.search(name):
            continue                                   # nunca los pesados fuente
        keep = False
        if ext in COPY_EXT:
            if ext == ".pdf" and "arrival" in rel and not args.include_arrival_pdf:
                keep = False
            else:
                keep = True
        elif ext == ".parquet":
            if any(k in name for k in COPY_PARQUET):
                keep = True
            elif name.startswith("df_final_new_trips") and args.include_df_final:
                keep = True
        if keep and ("/maps/" in f"/{rel}" or "/dashboards/" in f"/{rel}") and not args.include_maps:
            keep = False
        if keep and f.stat().st_size <= cap:
            picked.append(f)
    return picked


def write_route_zip(zpath: Path, rdir: Path, digest: dict, copy_files: list[Path],
                    interim_route: Path):
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        # digest computado
        for name, obj in (("route_digest.json", digest),
                          ("data_structure.json", digest.get("data_structure", {})),
                          ("trips_summary.json", digest.get("trips_summary", {})),
                          ("model_metrics.json", digest.get("model_metrics", {}))):
            zf.writestr(f"digest/{name}",
                        json.dumps(obj, ensure_ascii=False, indent=2, default=str))
        # artefactos copiados (conservando su ruta relativa dentro de interim)
        for f in copy_files:
            arc = "artifacts/" + str(f.relative_to(interim_route)).replace("\\", "/")
            try:
                zf.write(f, arc)
            except OSError:
                pass


# ── main por ruta ──────────────────────────────────────────────────

def discover_routes(root: Path) -> list[str]:
    base = root / "02_interim" / "routes"
    rs = set()
    if base.is_dir():
        rs |= {p.name for p in base.iterdir() if p.is_dir()}
    recs = root / "01_raw" / "records"
    if recs.is_dir():
        rs |= {p.name for p in recs.iterdir() if p.is_dir() and p.name.isdigit()}
    return sorted(rs, key=lambda r: (len(r), r))


def process_route(root: Path, route: str, args) -> dict:
    rdir = root / "02_interim" / "routes" / route
    t0 = time.time()
    digest: dict = {"route": route, "generated": datetime.now().isoformat(timespec="seconds")}
    digest["data_structure"] = build_data_structure(root, route)
    digest["trips_summary"] = build_trips_summary(rdir, args.variants) if rdir.is_dir() else {}
    digest["model_metrics"] = build_model_metrics(rdir, args.variants, args) if rdir.is_dir() else {}
    digest["headline"] = rollup_headline(route, digest["model_metrics"],
                                         digest["trips_summary"], digest["data_structure"])
    digest["_elapsed_s"] = r3(time.time() - t0)

    if args.mode in ("zip", "both"):
        copy_files = collect_copy_files(rdir, args)
        cbytes = sum(f.stat().st_size for f in copy_files)
        zpath = Path(args.out) / f"{route}.zip"
        write_route_zip(zpath, rdir, digest, copy_files, rdir)
        digest["_zip"] = {"path": str(zpath), "artifacts": len(copy_files),
                          "artifacts_bytes": cbytes,
                          "zip_size": human(zpath.stat().st_size)}
        log(f"  ✔ {route}.zip  ({len(copy_files)} artefactos, "
            f"{human(zpath.stat().st_size)})  [{digest['_elapsed_s']}s]")
    else:
        log(f"  ✔ digest {route}  [{digest['_elapsed_s']}s]")

    # Guarda el digest de ESTA ruta de inmediato → si el job se corta/timeout,
    # los resultados de las rutas ya hechas sobreviven.
    rdir_out = Path(args.out) / "routes"
    rdir_out.mkdir(parents=True, exist_ok=True)
    _atomic_write(rdir_out / f"{route}.json",
                  json.dumps(digest, ensure_ascii=False, indent=2, default=str))
    return digest


def _atomic_write(path: Path, text: str):
    """Escribe a un tmp y renombra: nunca deja un JSON a medias si se corta."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_global(out: Path, root: Path, variants, digests: dict, errors: dict) -> tuple[Path, Path]:
    glob = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "root": str(root),
        "variants": variants, "n_routes": len(digests),
        "errors": errors,
        "headline": {r: d.get("headline", {}) for r, d in digests.items()},
        "routes": digests,
    }
    payload = json.dumps(glob, ensure_ascii=False, indent=2, default=str)
    gpath = out / "analysis_digest.json"
    _atomic_write(gpath, payload)
    bundle = out / "digest_bundle.zip"
    btmp = bundle.with_suffix(".zip.tmp")
    with zipfile.ZipFile(btmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("analysis_digest.json", payload)
        for r, d in digests.items():
            zf.writestr(f"routes/{r}.json",
                        json.dumps(d, ensure_ascii=False, indent=2, default=str))
    os.replace(btmp, bundle)
    return gpath, bundle


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--routes", default="all")
    ap.add_argument("--variants", default="tmin,t1,t2")
    ap.add_argument("--mode", choices=["both", "digest", "zip"], default="both")
    ap.add_argument("-o", "--out", default="route_reports")
    ap.add_argument("--max-file-mb", type=float, default=60.0)
    ap.add_argument("--include-df-final", action="store_true")
    ap.add_argument("--include-arrival-pdf", action="store_true")
    ap.add_argument("--include-maps", action="store_true")
    ap.add_argument("--lstm-by-leg", action="store_true",
                    help="agrega también los df_errors del LSTM por leg (20-25 GB c/u; "
                         "redundante con vs_baselines; puede ser lento)")
    ap.add_argument("--quantile-max-gb", type=float, default=2.0,
                    help="omite p50/p90 (holísticas) en parquets mayores a esto (GB)")
    ap.add_argument("--max-agg-gb", type=float, default=0.0,
                    help="salta por completo la agregación de parquets mayores a esto "
                         "(GB); 0 = sin límite")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    root = Path(args.root).resolve() if args.root else find_root(Path.cwd())
    if root is None or not (root / "01_raw").is_dir():
        log("ERROR: no encuentro la raíz de datos (01_raw + 02_interim). Usa --root.")
        return 2
    routes = (discover_routes(root) if args.routes == "all"
              else [r.strip() for r in args.routes.split(",") if r.strip()])

    Path(args.out).mkdir(parents=True, exist_ok=True)
    log(f"root   : {root}")
    log(f"rutas  : {len(routes)} -> {routes}")
    log(f"modo   : {args.mode} | variantes {args.variants} | out {args.out}\n")

    out = Path(args.out)
    digests, errors = {}, {}
    gpath, bundle = write_global(out, root, args.variants, digests, errors)
    for i, route in enumerate(routes, 1):
        log(f"[{i}/{len(routes)}] ruta {route}")
        try:
            digests[route] = process_route(root, route, args)
        except Exception as e:
            errors[route] = f"{type(e).__name__}: {e}"
            log(f"  ✗ ERROR {route}: {e}\n{traceback.format_exc()[-800:]}")
        # Reescribe el digest global tras CADA ruta (barato, chico) → resultados
        # parciales siempre en disco aunque el job muera después.
        gpath, bundle = write_global(out, root, args.variants, digests, errors)

    log(f"\n=== LISTO ===")
    log(f"analysis_digest.json : {gpath}  ({human(gpath.stat().st_size)})")
    log(f"digest_bundle.zip    : {bundle}  ({human(bundle.stat().st_size)})  <- pásame ESTE")
    if args.mode in ("zip", "both"):
        zips = sorted(Path(args.out).glob("*.zip"))
        tot = sum(z.stat().st_size for z in zips if z.name != "digest_bundle.zip")
        log(f"zips por ruta        : {len(zips)-1} archivos, {human(tot)} en {args.out}")
    if errors:
        log(f"rutas con error      : {list(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
