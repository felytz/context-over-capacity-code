"""
run_new_pipeline.py
===================
Orquestador HEADLESS del MÉTODO NUEVO (viajes por leg, terminal→terminal) para
UNA ruta, de extremo a extremo: datos → viajes → features → df_full (con `leg`)
→ baselines (paso 10) → LSTM (paso 11) → XGBoost (paso 12).

Es el equivalente programático de los notebooks `new_pipeline_first.ipynb` +
`new_pipeline_after.ipynb`, pensado para correr VARIAS rutas sin abrir Jupyter
(p. ej. en bucle o desde la terminal). Replica el flujo ya validado en las rutas
1701 (1 semana) y 1602 (1 mes).

Diferencia clave con `run_all.py`: aquél usa el método VIEJO (dirección 0/1 vía
`make_dir_trips`/`turn_hex`); éste usa el método NUEVO (`_lib_new_trips.build_trips`
+ `assign_terminal_legs`), con la **leg** como unidad de análisis, lo que soporta
rutas con N terminales.

Uso (CLI)
---------
    # Ruta 1701, extracto de 1 semana en disco, ventana "rápida":
    python run_new_pipeline.py --route 1701 --date-range 01-07_2024-01 --mode rapido

    # Ruta 1602, un mes (custom):
    python run_new_pipeline.py --route 1602 \
        --date-range 2024_01-12 --mode custom \
        --start 2024-01-01 --end 2024-02-01

    # Sólo construir df_full (sin predicción):
    python run_new_pipeline.py --route 1701 --date-range 01-07_2024-01 \
        --mode rapido --no-baselines --no-lstm

    # VARIAS rutas (cada una en su subproceso; config es de un solo uso por
    # proceso). --date-range se omite → usa el extracto más grande por ruta.
    python run_new_pipeline.py --route 1602,1701 --mode rapido
    python run_new_pipeline.py --route all --exclude 1601 --mode rapido --zip

    # Nº de terminales (default 1). Cada valor crea un análisis APARTE, con la
    # etiqueta en el nombre (…_t1 / _t2 / _tauto): no se pisan entre sí.
    python run_new_pipeline.py --route all --terminals 1          # default
    python run_new_pipeline.py --route all --terminals 2          # otro análisis
    python run_new_pipeline.py --route all --terminals auto       # 1-2 auto/ruta

    # --zip deja data/03_processed/new_pipeline_<ruta>_<ventana>_<tN>.zip con
    # TODO el árbol de la ruta en 02_interim (mapas, dashboards, reportes,
    # df_errors, auditorías, perf/funnel) EXCEPTO dataframes/ y checkpoints.

    # En la nube: si en disco solo hay muestras parciales (la semana), bajar el
    # año completo de la BD (--prefer-full-range) y correr la ventana full.
    python run_new_pipeline.py --route all --mode full --prefer-full-range

Uso (import)
------------
    from run_new_pipeline import run_new_pipeline, run_routes
    res = run_new_pipeline(route=1701, date_range="01-07_2024-01", mode="rapido")
    res["df_full"]; res["baselines"]; res["lstm"]

    # Batch multi-ruta (subprocesos; devuelve resumen, no dataframes):
    resumen = run_routes("all", excluir=[1601], mode="rapido", make_zip=True)

Notas
-----
- El raw del extracto debe existir en
  `data/01_raw/records/<route>/df_rec_raw_<route>_<date_range>.parquet`
  (este script NO descarga de la BD).
- `df_full` NO trae `leg` desde el paso 07; aquí se une desde `df_final`
  (`assign_terminal_legs`) por `trip_id`, igual que la celda 6 del notebook
  `new_pipeline_after.ipynb`.
- Los viajes segmentados (build_trips + 3a) se CACHEAN en
  `dataframes/trips_cache_<ventana>/` y se reusan con `--no-force`; el manifest
  invalida el caché si cambian los parámetros del método o el nº de pings.
- Pasos 10/11 son group-aware: se les pasa `legs=[...]`; las legs con menos
  viajes que `n_folds` se SALTAN con aviso (InsufficientDataError), no abortan.
- El encoding UTF-8 de consola lo arregla `notebook_setup`/`config` al importarse.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

# Registra func/ en sys.path para correr desde cualquier cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Parámetros del método de viajes (los mismos validados en 1701/1602; ver
# _lib_new_trips.configure / docstring). Se exponen como constante para que
# sea fácil ajustarlos por ruta si hiciera falta.
_TRIP_PARAMS = dict(
    OFFROUTE_M=150.0, TERMINAL_NEAR_M=300.0, MIN_TRIP_COV=0.25, MIN_TRIP_PTS=5,
    TRIP_MARGIN=0.05, BREAK_MIN=10, GAP_SPLIT_MIN=20,
    HALF_LO=0.30, HALF_HI=0.70, JOIN_FULL_FRAC=0.90,
    ORPHAN_SPREAD_M=150.0, ORPHAN_DWELL_MIN=10.0,
)


def _render_pipeline_maps(BASE, ROUTE, df_rec_clean, df_stops, df_segments,
                          df_trips, poly, trip_meta, otab, lat0, lon0,
                          *, force=False, verbose=True):
    """Dibuja los mapas exploratorios del pipeline nuevo:
      - paso 2: route_overview / stopped_rec (self-cachean por is_file en step_02)
      - paso 3: new_trips / new_trips_orphans / new_trips_filtered (aquí se
        guardan por is_file salvo que ``force``).
    Todo va a ``BASE/maps/`` (categoría ``maps`` del zip). Best-effort: un mapa
    que falle avisa pero NO aborta el pipeline."""
    import _lib_new_trips as nt

    maps_dir = BASE / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    # paso 2 · overview + stopped. Se salta la llamada ENTERA si ambos HTML ya
    # existen y no se fuerza: build_overview_maps corre el detector turn_hex (caro)
    # ANTES de su guardia is_file interna, así que reusar solo es barato evitándolo.
    try:
        from config import PATHS as _P
        from step_02_route_overview import build_overview_maps
        _have = _P.overview_map.is_file() and _P.stopped_map.is_file()
        if _have and force:
            _P.overview_map.unlink(); _P.stopped_map.unlink()
            _have = False
        if _have:
            if verbose:
                print(f"  ✔ mapas overview/stopped existen: "
                      f"{_P.overview_map.name}, {_P.stopped_map.name} — reuso")
        else:
            _p = build_overview_maps(df_rec_clean, df_stops, df_segments,
                                     verbose=verbose)
            if verbose:
                for _k, _v in (_p or {}).items():
                    print(f"  ✔ mapa {_k}: {_v}")
    except Exception as e:
        print(f"  ⚠ mapas overview/stopped (paso 2) fallaron: {e}")

    def _skip(out):
        if out.is_file() and not force:
            if verbose:
                print(f"  ✔ mapa existe: {out.name} — reuso")
            return True
        return False

    # paso 3 · mapa de viajes (por bus, slider por viaje)
    try:
        out = maps_dir / f"new_trips_{ROUTE}.html"
        if not _skip(out):
            nt.render_trips_map(df_trips, poly, out, ROUTE, trip_meta)
            if verbose:
                print(f"  ✔ mapa viajes: {out}")
    except Exception as e:
        print(f"  ⚠ mapa new_trips falló: {e}")

    # paso 3 · mapa de huérfanos
    try:
        out = maps_dir / f"new_trips_orphans_{ROUTE}.html"
        if not _skip(out):
            nt.orphan_map(otab, poly, out, ROUTE, lat0, lon0)
            if verbose:
                print(f"  ✔ mapa huérfanos: {out}")
    except Exception as e:
        print(f"  ⚠ mapa orphans falló: {e}")

    # paso 3 · mapa filtrado (resalta depot_bound / start_dwell si existen)
    try:
        out = maps_dir / f"new_trips_filtered_{ROUTE}.html"
        if not _skip(out):
            sq = "depot_bound" if "depot_bound" in df_trips.columns else None
            sq2 = "start_dwell" if "start_dwell" in df_trips.columns else None
            nt.render_trips_map(df_trips, poly, out, ROUTE, trip_meta,
                                square_col=sq, square2_col=sq2, full_props=True)
            if verbose:
                print(f"  ✔ mapa filtrado: {out}")
    except Exception as e:
        print(f"  ⚠ mapa filtered falló: {e}")


def run_new_pipeline(
    *,
    route:        int,
    route_rec:    Optional[int] = None,
    date_range:   str,
    mode:         str = "rapido",          # full | rapido | custom
    start:        str = "2024-01-01",
    end:          str = "2024-02-01",
    force:        bool = True,
    work_hours_filter: bool = True,
    # viajes
    trip_cut:        str = "dwell",         # dwell | turnaround
    terminals:       object = 1,            # nº de terminales: int N | 'auto' | 'AUTO_min' | float
    layover_min_min: int = 10,
    min_leg_cov:     float = 0.4,
    # mapas
    do_maps:      bool = True,
    # predicción
    do_dashboards: bool = True,
    do_baselines: bool = True,
    do_lstm:      bool = True,
    do_xgb:       bool = False,
    # Solo entrenar legs DIRECCIONALES (inicio != fin). Los self-loop (T0->T0) son
    # residuo tras el fix del corte, no unidad de análisis → ~½ del paso 11 en basura.
    drop_self_loop_legs: bool = True,
    grid_step_m:  int = 100,
    n_folds:      int = 5,
    n_epochs:     int = 20,
    model_type:   str = "sequential",
    seed:         int = 42,
    # Régimen de la grid (pasos 10 y 12). 0.95/0.99 = histórico del paper; el
    # estudio pasó a 0.05/1.0 (soporte común). Ver docs/PLAN_REPRESENTATIVIDAD.md.
    n_route_quantile: float = 0.95,
    min_coverage:     float = 0.99,
    max_trips:    Optional[int] = None,
    xgb_exo:      bool = True,
    # empaquetado
    package_zip:  bool = False,
    verbose:      bool = True,
) -> Dict[str, object]:
    """Corre el método nuevo (leg) end-to-end para una ruta. Devuelve un dict con
    df_final, df_full, legs, y los resultados de los pasos 10 y 11 (o None)."""
    import time
    import numpy as np
    import polars as pl
    import notebook_setup as nbs

    route_rec = route if route_rec is None else route_rec
    t_all = time.time()

    # Pico de memoria (RSS) del proceso: getrusage lleva el máximo EXACTO que
    # tocó el proceso (mejor que el MaxRSS muestreado de sacct, que puede perder
    # spikes). Es monótono, así que imprimirlo por paso revela QUÉ paso empuja el
    # pico → sirve para dimensionar --mem por ruta en el reporte técnico.
    try:
        import resource as _resource

        def _peak_rss_gb():
            v = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss  # KiB (Linux) | bytes (macOS)
            v_kib = v / 1024 if sys.platform == "darwin" else v
            return v_kib / (1024 * 1024)
    except Exception:               # Windows/dev: sin resource → no reporta pico
        def _peak_rss_gb():
            return None

    # Banners de paso: "▶ inicio" / "✔ fin (Ns · pico RSS)". flush=True → salen en
    # vivo al log aun si el stdout está bufferizado (p. ej. .out de SLURM). Además
    # cada paso se ACUMULA en _perf_steps para el reporte técnico de rendimiento.
    _perf_steps: List[dict] = []
    _step_names: Dict[str, str] = {}

    def _s0(tag, name):
        _step_names[tag] = name
        if verbose:
            print(f"\n▶ PASO {tag} · {name} …", flush=True)
        return time.time()

    def _s1(tag, t0, extra=""):
        dt = time.time() - t0
        pk = _peak_rss_gb()
        _perf_steps.append({"step": tag, "name": _step_names.get(tag, ""),
                            "seconds": round(dt, 1),
                            "peak_rss_gb": (round(pk, 2) if pk is not None else None),
                            "note": (extra or None)})
        if verbose:
            _pk_s = f" · pico RSS {pk:.2f} GB" if pk is not None else ""
            print(f"✔ PASO {tag} listo en {dt:.1f}s"
                  f"{(' · ' + extra) if extra else ''}{_pk_s}", flush=True)

    # ── 0a/0b · ruta + ventana (fija env vars BUS_* e importa config) ─────────
    data_min, data_max = nbs.show_raw_extracts(route, date_range)
    config = nbs.configure_window(
        ruta=route, ruta_rec=route_rec, date_range=date_range, modo=mode,
        terminals=terminals,
        data_min=data_min, data_max=data_max,
        custom_start=start, custom_end=end,
    )
    PATHS = config.PATHS
    ROUTE = config.INT_ROUTE
    FULL_RANGE = config.ANALYSIS_RANGE     # ya incluye la etiqueta de terminales
    dominance_frac = config.DOMINANCE_FRAC  # fuente única (resuelta en configure_window)
    BASE = PATHS.interim_route_dir

    # ── 1 · carga ─────────────────────────────────────────────────────────────
    _t = _s0("1/12", "carga de datos (records + stops + segments)")
    from step_01_load_data import load_data
    df_rec_clean, df_stops, df_segments = load_data(
        verbose=verbose, force=force, work_hours_filter=work_hours_filter)
    _s1("1/12", _t, f"{df_rec_clean.height:,} pings")

    # ── 3 · viajes (método nuevo) ─────────────────────────────────────────────
    _t = _s0("3/12", "segmentación de viajes (build_trips + dirección/ciclo)")
    import _lib_new_trips as nt
    poly, cum, L = nt.load_route_polyline(str(PATHS.geojson_path), ROUTE)
    lat0 = float(np.mean([p[0] for p in poly]))
    lon0 = float(np.mean([p[1] for p in poly]))
    nt.configure(**_TRIP_PARAMS)

    # Caché de viajes: build_trips + 3a son deterministas dados los mismos pings
    # y parámetros, pero son el paso silencioso más caro (bucle Python por ping).
    # Con force=False se reusa lo persistido; el manifest invalida el caché si
    # cambian los parámetros o el nº de pings de entrada. Reusar también asegura
    # trip_id consistentes con el caché del enrich (misma sesión de viajes).
    _trips_cache_dir = BASE / "dataframes" / f"trips_cache_{FULL_RANGE}"
    # Terminales fijadas (TERMINAL_RULE=file): las de esta ruta en TERMINALS_FILE.
    _term_coords = nt.load_terminal_coords(ROUTE)
    if verbose and nt.TERMINAL_RULE == 'file':
        print(f"  terminales de {nt.TERMINALS_FILE}: "
              f"{_term_coords if _term_coords else 'la ruta no está → dwell'}", flush=True)
    _cache_params = dict(
        _TRIP_PARAMS, cap="v2", trip_cut=trip_cut, dominance_frac=dominance_frac,
        layover_min_min=layover_min_min, min_leg_cov=min_leg_cov,
        n_pings=df_rec_clean.height,
        # Versión del método: el manifest invalida por PARÁMETROS, no por cambios de
        # código, así que cada pieza que cambia el corte va en este marcador para que
        # --no-force recompute los viajes (nt.method_tag; docs/HANDOFF_METODO_V2.md).
        cut_fix=nt.method_tag(_term_coords))
    _cached = None if force else nt.load_trips_cache(
        _trips_cache_dir, _cache_params, verbose=verbose)
    if _cached is not None:
        df_trips, trip_meta, stats, otab, frag = _cached
    else:
        df_trips, trip_meta, stats, otab, frag = nt.build_trips(
            df_rec_clean, poly, cum, L, cap="v2",
            trip_cut=trip_cut, dominance_frac=dominance_frac,
            layover_min_min=layover_min_min, verbose=verbose,
            terminal_coords=_term_coords)

        # 3a · dirección geométrica + orden cíclico (plegado en trip_valid). v2 juzga el
        # ciclo por la secuencia de terminales, así que necesita las de las legs.
        _terms_3a = stats.get("terminals") or stats.get("zones", [])
        df_trips = nt.assign_geometric_direction(df_trips, poly, cum, L)
        df_trips = nt.mark_cyclic_order(df_trips, min_leg_cov=min_leg_cov,
                                        terminals=_terms_3a)
        df_trips = df_trips.with_columns(
            (pl.col("trip_valid") & pl.col("cyclic_ok")).alias("trip_valid"))
        stats["cyclic_order"] = nt.CYCLIC_ORDER
        stats["trips_valid_cyclic"] = int(
            df_trips.filter(pl.col("trip_valid"))["trip_id"].n_unique())
        if verbose:
            print(f"  orden cíclico ({nt.CYCLIC_ORDER}): {stats.get('trips_valid')} válidos → "
                  f"{stats['trips_valid_cyclic']}", flush=True)

        nt.save_trips_cache(_trips_cache_dir, _cache_params, df_trips, trip_meta,
                            stats, otab, frag, verbose=verbose)

    # Terminales que definen las LEGS: el set seleccionado por TERMINALES
    # (dominance_frac), no todas las zonas bearing. Ver build_trips.
    terminals = stats.get("terminals") or stats.get("zones", [])
    _s1("3/12", _t, f"{stats.get('trips_valid', '?')} viajes válidos · "
        f"{'cache' if _cached is not None else 'recalculado'}")

    # ── 4 · features df_final + leg ───────────────────────────────────────────
    _t = _s0("4/12", "features geométricas + asignación de legs (df_final)")
    base = (df_trips.filter(pl.col("trip_valid")).sort(["trip_id", "time"])
            .with_columns([pl.col("latitude").cast(pl.Float64),
                           pl.col("longitude").cast(pl.Float64)]))
    base = base.with_columns([
        (pl.col("latitude").diff().over("trip_id") * 111320).alias("dy"),
        (pl.col("longitude").diff().over("trip_id") * 111320 * np.cos(np.radians(lat0))).alias("dx")])
    base = base.with_columns(((pl.col("dx") ** 2 + pl.col("dy") ** 2).sqrt()).fill_null(0).alias("step_m"))
    base = base.with_columns([
        pl.col("step_m").cum_sum().over("trip_id").alias("cum_dist_m"),
        (pl.col("bin") / 100.0).alias("lin_pos")])
    df_final = nt.assign_terminal_legs(base, terminals, lat0=lat0)
    out_final = BASE / "dataframes" / f"df_final_new_trips_{FULL_RANGE}.parquet"
    out_final.parent.mkdir(parents=True, exist_ok=True)
    df_final.write_parquet(out_final)
    if verbose:
        print(f"  df_final: {df_final.shape} | terminales: {len(terminals)} | "
              f"legs: {sorted(df_final['leg'].unique().to_list())}")
    _s1("4/12", _t, f"{len(terminals)} terminales")

    # ── 4b · mapas (paso 2 overview/stopped + paso 3 viajes/huérfanos/filtrado) ─
    # Los que faltaban en el runner headless (el notebook new_pipeline_first sí los
    # dibuja). Cada mapa se salta si su HTML existe y no se fuerza; best-effort.
    if do_maps:
        _t = _s0("4b/12", "mapas (overview/stopped + viajes/huérfanos/filtrado)")
        _render_pipeline_maps(
            BASE, ROUTE, df_rec_clean, df_stops, df_segments,
            df_trips, poly, trip_meta, otab, lat0, lon0,
            force=force, verbose=verbose)
        _s1("4b/12", _t)

    # ── 5+7 · adaptar + enrich + features → df_full ───────────────────────────
    _t = _s0("5+7/12", "enrich + feature engineering (df_full)")
    df_trips_enrich = nt.prepare_trips_for_enrich(df_trips, poly, cum, L)

    from step_07_enrich_trips import run_enrich_step
    df_enriched, df_full = run_enrich_step(
        df_trips_enrich, df_segments, force=force, verbose=verbose)

    # leg vive en df_final (no en df_full); se une por trip_id (clave única por viaje)
    leg_map = df_final.select(
        ["trip_id", "leg"]).unique("trip_id")
    df_full = df_full.join(leg_map, on="trip_id", how="left")

    # GUARDIA: un `leg` null tras el join tiene DOS causas distintas y solo una
    # es un error. El discriminador correcto es la CONTENCIÓN de trip_id
    # (df_final ⊆ df_full), no el conteo de nulls:
    #  (a) STALE CACHE (error real): el enrich reusó shards de una SESIÓN
    #      ANTERIOR cuyos trip_id no coinciden con la segmentación actual → los
    #      viajes de df_final NO están en df_full y los pasos 10/11 correrían
    #      sobre datos equivocados. Detectado en la 1602 (209,658/240,165 nulls).
    #      Se re-enricha forzado con los viajes de ESTA sesión; si persiste, aborta.
    #  (b) SELF-LOOPS (esperado, NO error): df_full contiene viajes self-loop
    #      (T0->T0/T1->T1) que df_final excluye por diseño → su leg queda null,
    #      pero los trip_id de df_final SÍ están en df_full. El paso siguiente ya
    #      filtra los self-loops. Rutas sin layover en terminales quedan dominadas
    #      por estos (902: solo ~18 direccionales de 171); antes esto reventaba
    #      el job con un falso "mismatch".
    final_ids = set(df_final["trip_id"].unique().to_list())
    full_ids = set(df_full["trip_id"].unique().to_list())
    missing = final_ids - full_ids  # viajes de df_final ausentes en df_full = stale
    if missing and not force:
        frac = len(missing) / max(len(final_ids), 1)
        print(f"  ⚠ {len(missing):,}/{len(final_ids):,} viajes de df_final "
              f"({frac:.1%}) no están en df_full: el cache de enrich viene de otra "
              f"sesión. Re-enrich forzado …")
        df_enriched, df_full = run_enrich_step(
            df_trips_enrich, df_segments, force=True, verbose=verbose)
        df_full = df_full.join(leg_map, on="trip_id", how="left")
        full_ids = set(df_full["trip_id"].unique().to_list())
        missing = final_ids - full_ids
    if missing:
        frac = len(missing) / max(len(final_ids), 1)
        raise RuntimeError(
            f"{len(missing):,}/{len(final_ids):,} viajes de df_final ({frac:.1%}) "
            f"no están en df_full incluso tras re-enrich: no describen los mismos "
            f"viajes.")
    n_null_leg = int(df_full["leg"].null_count())
    if n_null_leg > 0:  # nulls remanentes = self-loops excluidos de df_final (ok)
        frac = n_null_leg / max(df_full.height, 1)
        print(f"  ℹ {n_null_leg:,} pings ({frac:.1%}) sin leg = viajes self-loop "
              f"fuera de df_final (esperado; se filtran en el paso siguiente).")

    # EJE DE LA MALLA (GRID_AXIS; oct 2026, docs/PREREGISTRO_VUELTA.md §2). Con el corte por
    # cruce y GRID_AXIS=traza, cum_dist_m pasa a ser el avance del bus sobre la traza desde el
    # primer registro del viaje (prog_m del df_final), no la suma de líneas rectas entre
    # registros que deja el enrich. Las rectas acortan las curvas: la malla del frame (P05 de la
    # longitud) cubría ~87 % de la vuelta y cada celda caía en otro sitio en cada viaje; sobre la
    # traza cubre ~98 % y cada celda es el mismo punto. Las features del enrich ya están
    # calculadas; cambian la malla, el objetivo y lo que el XGBoost y el LSTM muestrean por
    # posición (todos leen cum_dist_m del df_full).
    # Los viajes con un atajo (saltan kilómetros de traza) salen de trip_valid aquí, en el
    # df_full que leen los modelos: sus celdas saltadas no tienen tiempo observado. Siguen
    # válidos en el df_final y cuentan en el funnel. Con GRID_AXIS=recorrido (el default) no
    # cambia nada.
    _grid_axis = os.environ.get("GRID_AXIS", "recorrido").strip().lower()
    if _grid_axis not in ("recorrido", "traza"):
        raise ValueError(f"GRID_AXIS={_grid_axis!r}: va con recorrido | traza")
    if _grid_axis == "traza" and nt.TRIP_CUT != "cruce":
        raise ValueError("GRID_AXIS=traza necesita TRIP_CUT=cruce (prog_m sale de ese corte)")
    _n_atajo_fuera = 0
    if _grid_axis == "traza":
        _prog = (df_final.select(["trip_id", "time", "prog_m", "atajo"])
                 .with_columns(pl.col("atajo").any().over("trip_id").alias("_viaje_atajo"))
                 .drop("atajo"))
        df_full = df_full.join(_prog, on=["trip_id", "time"], how="left")
        _sin = df_full.filter(pl.col("trip_valid") & pl.col("prog_m").is_null())
        if _sin.height:
            raise RuntimeError(
                f"GRID_AXIS=traza: {_sin.height:,} registros de "
                f"{_sin['trip_id'].n_unique():,} viajes válidos del df_full no están en el "
                f"df_final (por trip_id y time): no hay prog_m con qué medir el avance.")
        _n_atajo_fuera = int(df_full.filter(pl.col("trip_valid") & pl.col("_viaje_atajo"))
                             ["trip_id"].n_unique())
        df_full = (df_full
                   .with_columns(
                       pl.when(pl.col("prog_m").is_not_null())
                         .then(pl.col("prog_m") - pl.col("prog_m").min().over("trip_id"))
                         .otherwise(pl.col("cum_dist_m"))
                         .alias("cum_dist_m"),
                       (pl.col("trip_valid") & ~pl.col("_viaje_atajo").fill_null(False))
                         .alias("trip_valid"))
                   .drop(["prog_m", "_viaje_atajo"]))
        if verbose:
            print(f"  eje de la malla: avance sobre la traza (GRID_AXIS=traza) · "
                  f"{_n_atajo_fuera:,} viajes con atajo fuera de los modelos", flush=True)

    # LEGS que definen los grupos de predicción (pasos 10 y 11). Por defecto solo las
    # DIRECCIONALES (inicio != fin): un `leg` self-loop (T0->T0) es un viaje que empieza
    # y termina en el MISMO terminal — residuo (parciales/abortados, ~4% tras el fix del
    # corte) y NO una unidad de análisis válida. Entrenarlos gastaba ~½ del paso 11 en
    # basura. Antes del fix eran la mayoría (99% en 902) y no se podían omitir.
    # Ver docs/HANDOFF_CORTE_GEOMETRICO.md. Incluirlos: --legs-self-loops.
    _all_legs: List[str] = sorted(df_full["leg"].drop_nulls().unique().to_list())
    def _is_directional(l: str) -> bool:
        p = str(l).split("->")
        return len(p) == 2 and p[0] != p[1]
    if nt.TRIP_CUT == "cruce":
        # Corte por cruce de km: todo viaje válido va de un cruce al siguiente, así que no
        # hay residuo. Con un solo punto de corte la leg es T0->T0 y ES la vuelta, la unidad
        # de análisis (Comparativo_metodos_viajes.md §10): no se omite.
        legs = _all_legs
        if verbose:
            print(f"  corte por cruce de km: legs {legs} (con un corte, T0->T0 = la vuelta)")
    elif drop_self_loop_legs:
        legs = [l for l in _all_legs if _is_directional(l)]
        _skipped = [l for l in _all_legs if l not in legs]
        if _skipped:
            _n_tot = df_full.height
            _n_sl = df_full.filter(pl.col("leg").is_in(_skipped)).height
            print(f"  ⊘ legs self-loop omitidas de predicción: {_skipped} "
                  f"({_n_sl:,}/{_n_tot:,} filas = {100 * _n_sl / max(_n_tot, 1):.1f}%)")
        if not legs:                      # ruta degenerada (N=1 / todo self-loop)
            print(f"  ⚠ NINGUNA leg direccional ({_all_legs}) → se usan todas "
                  f"(ruta no-compatible; ver dominio de aplicabilidad en el handoff)")
            legs = _all_legs
    else:
        legs = _all_legs
    df_full.write_parquet(str(BASE / "dataframes" / f"df_rec_full_{FULL_RANGE}.parquet"))
    if verbose:
        print(f"  df_full: {df_full.height:,} filas x {len(df_full.columns)} cols | legs: {legs}")
    _s1("5+7/12", _t, f"{df_full.height:,} filas · {len(legs)} legs")

    result: Dict[str, object] = {
        "config": config, "df_final": df_final, "df_full": df_full,
        "legs": legs, "terminals": terminals, "baselines": None, "lstm": None,
        "trajectories": None, "dashboards": None,
    }

    # ── funnel de datos por ruta (Tabla 1 del paper: crudo→limpio→viajes) ─────
    # Se escribe APENAS existe df_full para que sobreviva a un crash posterior
    # (LSTM/OOM). Fuente: raw en disco + stats de build_trips + df_final/df_full.
    try:
        import json as _json
        try:
            _n_raw = int(pl.scan_parquet(str(PATHS.df_rec_raw))
                         .select(pl.len()).collect().item())
        except Exception:
            _n_raw = None
        _trips_per_leg = {r["leg"]: int(r["n"]) for r in
                          df_final.group_by("leg")
                                  .agg(pl.col("trip_id").n_unique().alias("n"))
                                  .sort("leg").iter_rows(named=True)}
        # SELF-LOOPS: viajes que empiezan y terminan en el MISMO terminal (T0->T0).
        # Tras el fix del corte son residuo (parciales/abortados); antes dominaban
        # (99% en 902). `self_loop_pct` es LA métrica de salud del corte por ruta y
        # el criterio de compatibilidad. Ver docs/HANDOFF_CORTE_GEOMETRICO.md.
        _sl_legs = sorted(l for l in _trips_per_leg if not _is_directional(l))
        _dir_legs = sorted(l for l in _trips_per_leg if _is_directional(l))
        _n_sl = sum(_trips_per_leg[l] for l in _sl_legs)
        _n_dir = sum(_trips_per_leg[l] for l in _dir_legs)
        _n_legtot = _n_sl + _n_dir
        _funnel = {
            "route": ROUTE, "analysis_range": FULL_RANGE, "mode": mode,
            "date_range": date_range,
            # embudo de registros
            "pings_raw": _n_raw,
            "pings_clean": int(df_rec_clean.height),
            "pct_pings_kept": (round(100 * df_rec_clean.height / _n_raw, 2)
                               if _n_raw else None),
            "buses": stats.get("buses"), "bus_days": stats.get("bus_days"),
            # embudo de viajes (build_trips)
            "trips_turnaround_ref": stats.get("trips_orig"),
            "trips_recovered": stats.get("trips_rec"),
            "trips_valid": stats.get("trips_valid"),
            # tras el orden cíclico (3a): el paso que no tenía número en la Tabla S2
            "trips_valid_cyclic": stats.get("trips_valid_cyclic"),
            "cyclic_order": stats.get("cyclic_order"),
            "method": stats.get("method"),
            "terminal_fix": stats.get("terminal_fix"),
            "trips_merged_ge2": stats.get("merged_ge2"),
            "orphans": stats.get("orphans"),
            "orphans_inicio": stats.get("orph_inicio"),
            "orphans_fin": stats.get("orph_fin"),
            "orphans_inactivo": stats.get("orph_inactivo"),
            "bad_terminal": stats.get("bad_terminal"),
            "half_pairs": stats.get("half_pairs"),
            "off_route_pct": stats.get("off_pct"),
            "depot_run_pings": stats.get("depot_run_pings"),
            "depot_run_busdays": stats.get("depot_run_busdays"),
            "start_consistency": stats.get("start_consistency"),
            "end_consistency": stats.get("end_consistency"),
            # terminales / legs
            "trip_cut": stats.get("trip_cut"),
            "cruce": stats.get("cruce"),              # sólo con TRIP_CUT=cruce
            # eje de la malla y lo que entra a los modelos (GRID_AXIS=traza saca los
            # viajes con atajo del df_full; en df_final y en `cruce` siguen contando)
            "grid_axis": _grid_axis,
            "trips_atajo_out_of_models": _n_atajo_fuera,
            "trips_valid_models": int(df_full.filter(pl.col("trip_valid"))
                                      ["trip_id"].n_unique()),
            "dominance_frac": stats.get("dominance_frac"),
            "n_terminals": len(terminals),
            "n_legs": len(legs),
            "trips_per_leg": _trips_per_leg,
            # self-loops (salud del corte / dominio de aplicabilidad)
            "legs_all": sorted(_trips_per_leg),
            "legs_directional": _dir_legs,
            "legs_self_loop": _sl_legs,
            "trips_directional": _n_dir,
            "trips_self_loop": _n_sl,
            "self_loop_pct": (round(100 * _n_sl / _n_legtot, 2) if _n_legtot else None),
            "self_loop_trained": (not drop_self_loop_legs),
            "legs_predicted": list(legs),
            # dataset final
            "df_full_rows": int(df_full.height),
            "df_full_cols": len(df_full.columns),
        }
        _funnel_path = BASE / "perf" / f"funnel_{FULL_RANGE}.json"
        _funnel_path.parent.mkdir(parents=True, exist_ok=True)
        _funnel_path.write_text(_json.dumps(_funnel, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        result["funnel_path"] = str(_funnel_path)
        result["self_loop_pct"] = _funnel["self_loop_pct"]
        if verbose:
            print(f"  ✔ funnel de datos (Tabla 1): {_funnel_path.name}", flush=True)
            print(f"  ⤷ self-loops: {_n_sl:,}/{_n_legtot:,} viajes = "
                  f"{_funnel['self_loop_pct']}%  · direccionales: {_dir_legs} "
                  f"· self-loop: {_sl_legs or '—'}", flush=True)
    except Exception as _e:
        print(f"  ⚠ no pude escribir el funnel de datos: {_e}")

    # ── 8 · dashboards (por leg) ──────────────────────────────────────────────
    # El paso 8 (dashboards HTML) NO viaja en este repositorio: no produce nada
    # que el artículo use, y todas las corridas publicadas lo saltaron con
    # --no-dashboards (jobs/job_data_pipeline.slrm). El flag se conserva para que
    # esas invocaciones sigan siendo válidas tal cual.
    if do_dashboards:
        print("  ⊘ paso 8 (dashboards) no incluido en este repositorio — se omite")

    # ── 8b · trayectorias distancia–tiempo por leg (Fig. 2 del paper, con
    # panel sin outliers + clusters rápido/normal/lento). PNGs en reports/
    # (entran al zip). Es barato (numpy); si falla no aborta el pipeline.
    _t = _s0("8b/12", "trayectorias distancia–tiempo por leg")
    try:
        from _lib_plot import plot_time_distance_profiles
        result["trajectories"] = plot_time_distance_profiles(
            df_final, force=force, verbose=verbose)
    except Exception as e:
        print(f"  ⚠ trayectorias distancia–tiempo fallaron: {e}")
    _s1("8b/12", _t)

    # ── 10 · baselines por leg ────────────────────────────────────────────────
    if do_baselines:
        _t = _s0("10/12", f"baselines por leg ({len(legs)} legs, {n_folds}-fold CV, "
                 f"q={n_route_quantile} cov={min_coverage})")
        from step_10_arrival_prediction import run_arrival_prediction_step
        result["baselines"] = run_arrival_prediction_step(
            df_full, legs=legs, grid_step_m=grid_step_m, n_folds=n_folds,
            n_route_quantile=n_route_quantile, min_coverage=min_coverage,
            include_grid_500=False, force_rerun=force, verbose=verbose)
        _s1("10/12", _t)

    # ── 11 · LSTM por leg ─────────────────────────────────────────────────────
    if do_lstm:
        _t = _s0("11/12", f"LSTM por leg ({len(legs)} legs, {n_epochs} épocas, "
                 f"{n_folds}-fold CV) — el paso más caro")
        from step_11_lstm_prediction_tf import run_lstm_prediction_step
        result["lstm"] = run_lstm_prediction_step(
            df_full, legs=legs, model_type=model_type, n_epochs=n_epochs,
            n_folds=n_folds, seed=seed, force_rerun=force, verbose=verbose)
        _s1("11/12", _t)

    # ── 12 · XGBoost tabular por leg (CPU) ────────────────────────────────────
    # Corre en el MISMO loop de CV que los baselines y `delay_hour`, sobre las
    # MISMAS filas ⇒ la comparación es pareada por construcción. Además escribe
    # el FRAME de evaluación (`func/eval_frame.py`) que consume el LSTM.
    if do_xgb:
        _t = _s0("12/12", f"XGBoost por leg ({len(legs)} legs, "
                 f"tier={'A+B' if xgb_exo else 'A'}, q={n_route_quantile})")
        try:
            from xgb_prediction.run_experiment import run_one_leg as _xgb_leg
            from xgb_prediction.run_experiment import persist as _xgb_persist
            from xgb_prediction.run_experiment import _out_dir as _xgb_out_dir
            out_dir_xgb = _xgb_out_dir()
            xgb_res = {}
            for leg in legs:
                r = _xgb_leg(
                    df_full, leg, grid_step_m=grid_step_m, n_folds=n_folds,
                    seed=seed, n_route_quantile=n_route_quantile,
                    min_coverage=min_coverage, max_trips=max_trips,
                    with_exo=xgb_exo, verbose=verbose,
                )
                if r is None:
                    continue
                _xgb_persist(r, out_dir_xgb, verbose=verbose)
                xgb_res[leg] = r["overall"]
            result["xgb"] = xgb_res
        except ImportError as e:
            # xgboost/scikit-learn son dependencias NUEVAS del paso 12; que falten
            # no debe tumbar una corrida cuyo objetivo eran los pasos 10/11.
            print(f"  ⚠ paso 12 (XGBoost) saltado: {e}")
            print(f"    instala con: pip install 'xgboost>=2.0' 'scikit-learn>=1.3'")
            result["xgb"] = None
        _s1("12/12", _t)

    # ── 📦 · zip de outputs relevantes en data/03_processed ──────────────────
    if package_zip:
        _t = _s0("zip", "empaquetado de outputs en data/03_processed")
        from _lib_package import package_route_outputs
        result["zip"] = package_route_outputs(PATHS, verbose=verbose)
        _s1("zip", _t)

    _peak = _peak_rss_gb()
    _elapsed_min = round((time.time() - t_all) / 60.0, 2)
    result["peak_rss_gb"] = _peak
    result["elapsed_min"] = _elapsed_min

    # --mem sugerido: pico ×1.2 de holgura (el cgroup de SLURM cuenta también
    # page-cache), al escalón estándar más cercano hacia arriba.
    _tiers = [32, 48, 64, 96, 128]
    if _peak is not None:
        _need = _peak * 1.2
        _fits = next((t for t in _tiers if _need <= t), None)
        _mem_sug = f"{_fits}G" if _fits else f">{_tiers[-1]}G"
    else:
        _mem_sug = None

    # ── reporte técnico de rendimiento — SIEMPRE a disco (para la literatura) ──
    # perf/<ventana>.json por ruta: tiempo, pico RSS, --mem sugerido y desglose
    # por paso. _lib_perf.collect_perf_reports los consolida en una tabla.
    try:
        import json as _json, socket as _socket, datetime as _dtmod
        _perf_record = {
            "route": ROUTE, "analysis_range": FULL_RANGE, "mode": mode,
            "finished_utc": _dtmod.datetime.now(_dtmod.timezone.utc)
                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hostname": _socket.gethostname(),
            "n_cpus": int(os.environ.get("SLURM_CPUS_PER_TASK",
                                         os.cpu_count() or 0) or 0),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "force": force,
            "steps_ran": {"maps": do_maps, "dashboards": do_dashboards,
                          "baselines": do_baselines, "lstm": do_lstm},
            "n_legs": len(legs), "n_trips_valid": stats.get("trips_valid"),
            "df_full_rows": int(df_full.height),
            "elapsed_min": _elapsed_min,
            "peak_rss_gb": (round(_peak, 2) if _peak is not None else None),
            "suggested_mem_g": _mem_sug,
            "steps": _perf_steps,
        }
        _perf_path = BASE / "perf" / f"perf_{FULL_RANGE}.json"
        _perf_path.parent.mkdir(parents=True, exist_ok=True)
        _perf_path.write_text(_json.dumps(_perf_record, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        result["perf_path"] = str(_perf_path)
    except Exception as _e:
        print(f"  ⚠ no pude escribir el reporte de rendimiento: {_e}")

    if verbose:
        print(f"\n  ✅ RUN_NEW_PIPELINE completo · ruta {ROUTE} · {FULL_RANGE} · legs={legs}")
        # Resumen grep-able: 'RECURSOS ruta'. El pico de RSS es el máximo exacto
        # del proceso (getrusage); cross-check: sacct -j <jobid> --format=MaxRSS.
        print("  " + "═" * 60)
        print(f"  RECURSOS ruta {ROUTE} · {FULL_RANGE}")
        print(f"    tiempo total     : {_elapsed_min:.1f} min")
        if _peak is not None:
            print(f"    pico memoria RSS : {_peak:.2f} GB")
            print(f"    --mem sugerido   : {_mem_sug}  (pico ×1.2 de holgura)")
        print(f"    reporte          : {result.get('perf_path', '(no escrito)')}")
        print("  " + "═" * 60, flush=True)
    return result


# ──────────────────────────────────────────────────────────────────
# BATCH MULTI-RUTA (subprocesos)
# ──────────────────────────────────────────────────────────────────

def run_routes(
    rutas="all",
    excluir=None,
    date_range=None,           # None → extracto más grande por ruta | str | dict
    *,
    prefer_full_range: bool = False,   # solo muestra parcial en disco → baja el año completo de la BD
    full_range:   str = "2024_01-12",
    mode:         str = "rapido",
    terminals:    object = 1,
    start:        str = "2024-01-01",
    end:          str = "2024-02-01",
    force:        bool = True,
    do_maps:      bool = True,
    do_dashboards: bool = True,
    do_baselines: bool = True,
    do_lstm:      bool = True,
    do_xgb:       bool = False,
    drop_self_loop_legs: bool = True,
    grid_step_m:  int = 100,
    n_folds:      int = 5,
    n_epochs:     int = 20,
    seed:         int = 42,
    n_route_quantile: float = 0.95,
    min_coverage:     float = 0.99,
    max_trips:    Optional[int] = None,
    xgb_exo:      bool = True,
    make_zip:     bool = False,
    verbose:      bool = True,
) -> List[Dict[str, object]]:
    """Corre el pipeline nuevo para VARIAS rutas, cada una en un SUBPROCESO.

    Necesario porque ``config`` se parametriza vía env vars una sola vez por
    proceso (el guard de ``configure_window`` aborta si se re-parametriza);
    un subproceso por ruta garantiza aislamiento total, también desde un
    kernel de Jupyter que ya tenga ``config`` importado para otra ruta.

    ``rutas``/``excluir``/``date_range`` — misma semántica que
    ``notebook_setup.resolve_routes`` (int | lista | "all"; excluir lista;
    date_range None=auto | str | dict por ruta).

    Devuelve un resumen por ruta: ``[{route, date_range, ok, returncode,
    seconds}, ...]`` (NO dataframes; para eso usa ``run_new_pipeline`` con
    una sola ruta).
    """
    import subprocess
    import time

    import notebook_setup as nbs

    plan = nbs.resolve_routes(rutas, excluir=excluir, date_range=date_range,
                              prefer_full_range=prefer_full_range,
                              full_range=full_range, verbose=verbose)

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "run_new_pipeline.py")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    # Limpia BUS_* heredados (p. ej. de un configure_window previo en este
    # proceso) para que cada subproceso parta de cero.
    for k in list(env):
        if k.startswith("BUS_"):
            env.pop(k)

    summary: List[Dict[str, object]] = []
    for i, (route, dr) in enumerate(plan, 1):
        cmd = [sys.executable, script,
               "--route", str(route), "--date-range", dr, "--mode", mode,
               "--terminals", str(terminals),
               "--start", start, "--end", end,
               "--grid-step-m", str(grid_step_m), "--n-folds", str(n_folds),
               "--n-epochs", str(n_epochs), "--seed", str(seed),
               "--n-route-quantile", str(n_route_quantile),
               "--min-coverage", str(min_coverage)]
        if max_trips is not None:
            cmd += ["--max-trips", str(max_trips)]
        if do_xgb:
            cmd.append("--xgb")
        if not xgb_exo:
            cmd.append("--no-xgb-exo")
        if not force:
            cmd.append("--no-force")
        if not do_maps:
            cmd.append("--no-maps")
        if not do_dashboards:
            cmd.append("--no-dashboards")
        if not do_baselines:
            cmd.append("--no-baselines")
        if not do_lstm:
            cmd.append("--no-lstm")
        if not drop_self_loop_legs:
            cmd.append("--legs-self-loops")
        if make_zip:
            cmd.append("--zip")

        print(f"\n{'=' * 64}\n  [{i}/{len(plan)}] RUTA {route} · '{dr}' · mode={mode}"
              f"\n{'=' * 64}")
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=env)
        for line in proc.stdout:
            print(line, end="")
        rc = proc.wait()
        dt = time.time() - t0
        ok = rc == 0
        print(f"\n  {'✅' if ok else '❌'} ruta {route}: "
              f"{'OK' if ok else f'FALLÓ (rc={rc})'} en {dt / 60:.1f} min")
        summary.append({"route": route, "date_range": dr, "ok": ok,
                        "returncode": rc, "seconds": round(dt, 1)})

    n_ok = sum(1 for s in summary if s["ok"])
    print(f"\n{'=' * 64}\n  BATCH terminado: {n_ok}/{len(summary)} rutas OK")
    for s in summary:
        print(f"    {'✅' if s['ok'] else '❌'} {s['route']}  '{s['date_range']}'  "
              f"{s['seconds'] / 60:.1f} min")
    return summary


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Pipeline método nuevo (leg) para una o varias rutas.")
    p.add_argument("--route", required=True,
                   help="una ruta (1602), lista separada por comas (1602,1701) "
                        "o 'all' (todas las que tengan extracto raw)")
    p.add_argument("--exclude", default="",
                   help="rutas a excluir, separadas por comas (útil con --route all)")
    p.add_argument("--route-rec", type=int, default=None,
                   help="group_id de BD del extracto (default = --route; "
                        "solo aplica con UNA ruta)")
    p.add_argument("--date-range", default=None,
                   help="etiqueta del extracto raw, p. ej. 01-07_2024-01 o "
                        "2024_01-12 (omitido → extracto más grande por ruta)")
    p.add_argument("--prefer-full-range", dest="prefer_full_range",
                   action="store_true",
                   help="si en disco solo hay muestras parciales (p. ej. la "
                        "semana 01-07_2024-01), apunta al año completo y deja "
                        "que el paso 01 lo DESCARGUE de la BD (requiere BD; "
                        "úsalo en la nube)")
    p.add_argument("--full-range", default="2024_01-12",
                   help="etiqueta del extracto completo a descargar con "
                        "--prefer-full-range (default 2024_01-12)")
    p.add_argument("--mode", choices=["full", "rapido", "custom"], default="rapido")
    p.add_argument("--terminals", default="1",
                   help="nº de terminales del método de viajes: 1, 2, … | 'auto' | "
                        "'AUTO_min' | float (fracción). Etiqueta el análisis "
                        "(…_t1/_t2/_tauto) para no pisar corridas previas")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-02-01")
    p.add_argument("--no-force", dest="force", action="store_false")
    p.add_argument("--no-maps", dest="do_maps", action="store_false",
                   help="salta los mapas (overview/stopped del paso 2 + "
                        "viajes/huérfanos/filtrado del paso 3)")
    p.add_argument("--no-dashboards", dest="do_dashboards", action="store_false",
                   help="salta los dashboards (paso 8)")
    p.add_argument("--no-baselines", dest="do_baselines", action="store_false")
    p.add_argument("--no-lstm", dest="do_lstm", action="store_false")
    p.add_argument("--xgb", dest="do_xgb", action="store_true",
                   help="corre el paso 12 (XGBoost tabular, CPU) en el MISMO loop "
                        "de CV que los baselines. Además escribe el frame de "
                        "evaluación que consume el LSTM (func/eval_frame.py).")
    p.add_argument("--no-xgb-exo", dest="xgb_exo", action="store_false",
                   help="XGBoost solo Tier A (sin exógenos ni delay_hour)")
    p.add_argument("--legs-self-loops", dest="drop_self_loop_legs",
                   action="store_false",
                   help="entrena TAMBIÉN las legs self-loop (T0->T0); por default se "
                        "omiten (residuo del corte, no unidad de análisis, ~1/2 del paso 11)")
    p.add_argument("--zip", dest="package_zip", action="store_true",
                   help="empaqueta los outputs relevantes en data/03_processed/")
    p.add_argument("--grid-step-m", type=int, default=100)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-route-quantile", type=float, default=0.95,
                   help="cuantil de longitud que fija la extensión de la grid. "
                        "0.95 = histórico (retiene ~5%% de los viajes, los MÁS "
                        "LARGOS); 0.05 = soporte común (~95%%). Ver "
                        "docs/PLAN_REPRESENTATIVIDAD.md")
    p.add_argument("--min-coverage", type=float, default=0.99,
                   help="cobertura mínima de la grid. 1.0 elimina la saturación "
                        "de np.interp (tiempo restante fabricado) y da paridad "
                        "exacta de n_pairs con el LSTM")
    p.add_argument("--max-trips", type=int, default=None,
                   help="tope de viajes por leg en el paso 12 (KNN/kernel son "
                        "O(M²·l): con q=0.05 su coste se multiplica por ~200)")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    _excl = [int(x) for x in a.exclude.split(",") if x.strip()]
    _is_multi = (a.route.strip().lower() == "all") or ("," in a.route)

    if _is_multi:
        rutas = "all" if a.route.strip().lower() == "all" else \
            [int(x) for x in a.route.split(",") if x.strip()]
        run_routes(
            rutas, excluir=_excl, date_range=a.date_range,
            prefer_full_range=a.prefer_full_range, full_range=a.full_range,
            mode=a.mode, terminals=a.terminals,
            start=a.start, end=a.end, force=a.force,
            do_maps=a.do_maps,
            do_dashboards=a.do_dashboards, do_baselines=a.do_baselines,
            do_lstm=a.do_lstm, do_xgb=a.do_xgb,
            drop_self_loop_legs=a.drop_self_loop_legs,
            grid_step_m=a.grid_step_m, n_folds=a.n_folds, n_epochs=a.n_epochs,
            seed=a.seed, make_zip=a.package_zip,
            n_route_quantile=a.n_route_quantile, min_coverage=a.min_coverage,
            max_trips=a.max_trips, xgb_exo=a.xgb_exo,
        )
    else:
        route = int(a.route)
        date_range = a.date_range
        if date_range is None:
            import notebook_setup as _nbs
            [(route, date_range)] = _nbs.resolve_routes(
                route, excluir=_excl,
                prefer_full_range=a.prefer_full_range, full_range=a.full_range)
        run_new_pipeline(
            route=route, route_rec=a.route_rec, date_range=date_range,
            mode=a.mode, terminals=a.terminals,
            start=a.start, end=a.end, force=a.force,
            do_maps=a.do_maps,
            do_dashboards=a.do_dashboards, do_baselines=a.do_baselines,
            do_lstm=a.do_lstm, do_xgb=a.do_xgb,
            drop_self_loop_legs=a.drop_self_loop_legs,
            grid_step_m=a.grid_step_m, n_folds=a.n_folds, n_epochs=a.n_epochs,
            seed=a.seed, package_zip=a.package_zip,
            n_route_quantile=a.n_route_quantile, min_coverage=a.min_coverage,
            max_trips=a.max_trips, xgb_exo=a.xgb_exo,
        )
