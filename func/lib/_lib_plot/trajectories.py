"""
_lib_plot.trajectories
======================
Trayectorias distancia–tiempo por leg (extensión de la Fig. 2 del paper de
Dublin Bus 46A): cada viaje es una curva que va del 0 al 100 % de la ruta
(eje X en km) contra el tiempo transcurrido desde el inicio del viaje (eje Y
en minutos). Permite ver de un vistazo cuánto suelen tardar los viajes y cómo
se abre el abanico a lo largo de la ruta.

Por cada leg se genera UNA figura PNG con 5 paneles:

  A0 · Sin filtros — TODOS los viajes válidos tal cual salen de df_final
      (sin interpolar, sin exigir cobertura), como referencia cruda.
  A · Filtro leve   — viajes que pasan el filtro estructural de la matriz de
      trayectorias (≥5 puntos, cobertura ≥ ``min_coverage``, distancia
      monótona), con banda P5/mediana/P95.
  B · Sin outliers  — se excluyen los viajes cuya DURACIÓN TOTAL cae fuera de
      la cerca de Tukey (Q1−k·IQR, Q3+k·IQR) para apreciar el comportamiento
      "usual".
  C · Clusters      — k-means (numpy, sin sklearn) sobre el PERFIL completo
      distancia–tiempo del subconjunto sin outliers, k=3 → rápido / normal /
      lento, con la mediana de cada cluster resaltada.
  D · Clusters por FORMA — k-means sobre el perfil NORMALIZADO por la duración
      total (cada viaje va de 0 a 100 % de su propio tiempo): separa por DÓNDE
      se gasta el tiempo, no por cuánto. Con k=3 los clusters se etiquetan
      "lento al inicio" / "uniforme" / "lento al final" según la fracción del
      tiempo consumida en la primera mitad de la ruta.

PAUSAS ("bumps"): sobre el perfil mediano del subconjunto usual se detectan
los tramos donde el ritmo (min/km) supera ``bump_pace_factor`` veces el ritmo
mediano de la ruta — el escalón típico de una espera/base a media ruta. Cada
pausa se marca con línea vertical en todos los paneles y se GEOLOCALIZA
tomando la mediana de lat/lon de los registros de df_final que caen en ese
punto del recorrido (el bus emite muchos pings parado ahí, así que la mediana
cae en el lugar de la espera).

HTML interactivo: además del PNG se escribe un ``.html`` autocontenido y
LIVIANO por leg (bandas de percentiles, medianas de clusters y una muestra
decimada de viajes — nunca los cientos de miles de pings), con selector de
panel, crosshair, pausas clicables con link a Google Maps e histograma de
duraciones. Ver ``_traj_html.py``.

Reusa ``arrival_prediction.dataset_builder.build_trajectory_matrix`` (la misma
interpolación en grid uniforme que usa el paso 10), así que los paneles son
consistentes con las matrices que alimentan los baselines.

Los PNG/HTML van a ``PATHS.reports_dir`` (``trajectories_leg<...>_<ventana>``),
que es una de las categorías que empaqueta ``_lib_package`` al zip de
``data/03_processed``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

__all__ = ["plot_time_distance_profiles"]

# Colores consistentes con arrival_prediction.plots.plot_trajectories
# (P5 azul = rápido, mediana negra, P95 rojo = lento).
_CLUSTER_STYLE = {
    "rápido": "#0275d8",
    "normal": "#5cb85c",
    "lento":  "#d9534f",
}

# Panel D (forma): ordenados por fracción del tiempo en la 1ª mitad de la ruta.
_SHAPE_STYLE = {
    "lento al inicio": "#e67e22",
    "uniforme":        "#5cb85c",
    "lento al final":  "#8e44ad",
}

_BUMP_COLOR = "#c0392b"


def _kmeans(X: np.ndarray, k: int, seed: int = 0, n_iter: int = 200):
    """k-means con init k-means++ (numpy puro; el venv no trae sklearn)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    # k-means++: primer centro al azar, siguientes ∝ distancia² al más cercano
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
        p = d2 / d2.sum() if d2.sum() > 0 else None
        centers.append(X[rng.choice(n, p=p)])
    C = np.array(centers)

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)   # (n, k)
        new_labels = d.argmin(1)
        newC = np.array([X[new_labels == j].mean(0) if np.any(new_labels == j)
                         else C[j] for j in range(k)])
        if np.array_equal(new_labels, labels) and np.allclose(newC, C):
            break
        labels, C = new_labels, newC
    return labels, C


def _draw_band(ax, x_km, T_min, n_background, seed):
    """Fondo de trayectorias + banda P5/mediana/P95 (estilo Fig. 2)."""
    rng = np.random.default_rng(seed)
    M = T_min.shape[0]
    for i in rng.choice(M, size=min(n_background, M), replace=False):
        ax.plot(x_km, T_min[i], color="#aaaaaa", lw=0.35, alpha=0.35)
    ax.plot(x_km, np.percentile(T_min, 5, axis=0), color="#0275d8", lw=2.0,
            label="Rápido (P5)", zorder=5)
    ax.plot(x_km, np.percentile(T_min, 50, axis=0), color="black", lw=2.4,
            label="Mediano", zorder=5)
    ax.plot(x_km, np.percentile(T_min, 95, axis=0), color="#d9534f", lw=2.0,
            label="Lento (P95)", zorder=5)
    ax.legend(loc="upper left", fontsize=9)


# ────────────────────────────────────────────────────────────────────
# Pausas ("bumps") en el perfil mediano
# ────────────────────────────────────────────────────────────────────

def _detect_bumps(t50: np.ndarray, x_km: np.ndarray, *,
                  pace_factor: float = 2.0, min_extra_min: float = 0.8,
                  exclude_frac: float = 0.05, max_bumps: int = 3) -> List[dict]:
    """Tramos donde el ritmo mediano (min/km) supera ``pace_factor`` veces el
    ritmo mediano global — el "escalón" de una pausa usual a media ruta.
    Excluye los extremos (``exclude_frac`` de cada lado: ahí viven las esperas
    de terminal, que no son pausas de recorrido)."""
    n = len(x_km)
    if n < 10:
        return []
    pace = np.gradient(t50, x_km)                        # min/km
    i0, i1 = int(n * exclude_frac), n - int(n * exclude_frac)
    base = float(np.median(pace[i0:i1]))
    if base <= 0:
        return []
    mask = np.zeros(n, dtype=bool)
    mask[i0:i1] = pace[i0:i1] > pace_factor * base

    bumps: List[dict] = []
    i = i0
    while i < i1:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < i1 and mask[j + 1]:
            j += 1
        a, b = max(i - 1, 0), min(j + 1, n - 1)          # ± un punto de grid
        extra = float((t50[b] - t50[a]) - base * (x_km[b] - x_km[a]))
        if extra >= min_extra_min:
            peak = a + int(np.argmax(pace[a:b + 1]))
            bumps.append({
                "km":        float(x_km[peak]),
                "from_km":   float(x_km[a]),
                "to_km":     float(x_km[b]),
                "i_from":    a, "i_to": b,
                "extra_min": round(extra, 1),
                "base_pace": round(base, 2),
            })
        i = j + 1
    bumps.sort(key=lambda d: -d["extra_min"])
    return bumps[:max_bumps]


def _bump_trip_stats(bump: dict, T_usual: np.ndarray, x_km: np.ndarray) -> None:
    """Enriquece el bump con estadísticas POR VIAJE: cuánto tiempo extra gasta
    cada viaje al cruzar el tramo (vs el ritmo base) y qué % de viajes pausan."""
    a, b = bump["i_from"], bump["i_to"]
    cross = T_usual[:, b] - T_usual[:, a]                            # min
    expected = bump["base_pace"] * (x_km[b] - x_km[a])
    extra = cross - expected
    bump["extra_min_median_trip"] = round(float(np.median(extra)), 1)
    bump["pct_trips_pausing"] = round(float(np.mean(extra > 1.0)) * 100, 0)


def _geo_at_km(df_leg, km: float, window_m: float = 150.0):
    """Mediana de lat/lon de los registros que caen a ``±window_m`` de ese punto
    del recorrido. Los pings emitidos durante la pausa dominan la mediana, así
    que el resultado apunta al lugar de la espera."""
    import polars as pl
    sub = df_leg.filter((pl.col("cum_dist_m") - km * 1000).abs() <= window_m)
    if sub.height == 0:
        return None
    return (round(float(sub["latitude"].median()), 5),
            round(float(sub["longitude"].median()), 5))


# ────────────────────────────────────────────────────────────────────
# Figura principal
# ────────────────────────────────────────────────────────────────────

def plot_time_distance_profiles(
    df_final,
    legs: Optional[list] = None,
    *,
    group_col:        str = "leg",
    grid_step_m:      int = 100,
    min_coverage:     float = 0.90,     # filtro LEVE del panel A
    n_route_quantile: float = 0.95,
    outlier_k:        float = 1.5,      # cerca de Tukey (k·IQR) sobre duración
    k_clusters:       int = 3,
    n_background:     int = 400,
    detect_bumps:     bool = True,
    bump_pace_factor: float = 2.0,
    bump_min_extra_min: float = 0.8,
    make_html:        bool = True,
    out_dir:          Optional[Path] = None,
    suffix:           Optional[str] = None,
    seed:             int = 0,
    show:             bool = False,
    force:            bool = False,
    verbose:          bool = True,
) -> Dict[str, dict]:
    """Genera la figura distancia–tiempo (5 paneles) por leg: PNG + HTML.

    Parameters
    ----------
    df_final : pl.DataFrame
        Salida del método nuevo (necesita bus_code, trip_id, time, cum_dist_m,
        latitude, longitude, trip_valid y ``group_col``).
    legs : list | None
        Legs a graficar; None → todas las presentes en ``df_final``.
    min_coverage : float
        Cobertura mínima de la ruta para entrar al panel A ("filtro leve").
    outlier_k : float
        Multiplicador de la cerca de Tukey sobre la duración total (panel B).
    k_clusters : int
        Clusters de los paneles C/D (3 → rápido/normal/lento y lento al
        inicio/uniforme/lento al final). Se saltan si quedan menos de ``4·k``
        viajes sin outliers.
    detect_bumps / bump_pace_factor / bump_min_extra_min :
        Detección de pausas usuales a media ruta sobre el perfil mediano
        (ritmo > factor × ritmo base; extra mínimo en minutos). Cada pausa se
        marca con línea vertical y se geolocaliza (mediana lat/lon de los
        registros en ese punto del recorrido).
    make_html : bool
        Además del PNG escribe un HTML interactivo autocontenido (liviano:
        bandas + medianas + muestra decimada, no los pings crudos).
    out_dir / suffix : destino; defaults ``PATHS.reports_dir`` y
        ``config.ANALYSIS_RANGE``.
    show : bool
        True → deja las figuras abiertas (para que Jupyter las muestre);
        False → las cierra tras guardar (modo headless).

    Returns
    -------
    dict {leg: {png, html, n_trips, n_outliers, clusters, shape_clusters,
                bumps, fig|None}}
    """
    import matplotlib.pyplot as plt
    import polars as pl

    from _lib_style import apply_style
    from arrival_prediction.dataset_builder import build_trajectory_matrix

    apply_style()

    if out_dir is None or suffix is None:
        import config
        out_dir = Path(config.PATHS.reports_dir) if out_dir is None else Path(out_dir)
        suffix = config.ANALYSIS_RANGE if suffix is None else suffix
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if legs is None:
        legs = sorted(df_final.filter(pl.col("trip_valid"))[group_col]
                      .drop_nulls().unique().to_list())

    results: Dict[str, dict] = {}
    for leg in legs:
        # ── caché: si el PNG (y el HTML, si aplica) ya existen y no se fuerza,
        # se salta TODO el cómputo caro (matriz + k-means + render) de esta leg ─
        safe = re.sub(r"[^0-9A-Za-z_]+", "", str(leg).replace("->", "_"))
        png = out_dir / f"trajectories_leg{safe}_{suffix}.png"
        html_path = out_dir / f"trajectories_leg{safe}_{suffix}.html"
        if not force and png.is_file() and (not make_html or html_path.is_file()):
            if verbose:
                print(f"  ✔ {group_col}={leg}: trayectorias ya existen "
                      f"({png.name}) — reuso")
            results[str(leg)] = {"png": png,
                                 "html": html_path if make_html else None,
                                 "cached": True}
            continue

        # ── registros crudos de la leg (panel A0 + geolocalización de pausas) ─
        df_leg = (df_final
                  .filter(pl.col("trip_valid") & (pl.col(group_col) == leg))
                  .sort(["trip_id", "time"])
                  .with_columns(
                      ((pl.col("time") - pl.col("time").min().over("trip_id"))
                       .dt.total_seconds() / 60.0).alias("t_min"),
                      (pl.col("cum_dist_m") / 1000.0).alias("d_km")))
        n_raw = df_leg["trip_id"].n_unique()

        # ── matriz de trayectorias (misma interpolación que el paso 10) ──
        try:
            built = build_trajectory_matrix(
                df_final, group_col=group_col, group_value=leg,
                grid_step_m=grid_step_m, n_route_quantile=n_route_quantile,
                min_coverage=min_coverage, verbose=False)
        except ValueError as e:
            if verbose:
                print(f"  ⚠ {group_col}={leg}: {e}")
            continue

        T = built["T"]                      # (M, n) segundos
        x_km = built["l_grid"] / 1000.0
        T_min = T / 60.0
        M = T.shape[0]

        # ── outliers por duración total (cerca de Tukey) ──
        total = T_min[:, -1]
        q1, q3 = np.percentile(total, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - outlier_k * iqr, q3 + outlier_k * iqr
        keep = (total >= lo) & (total <= hi)
        T_usual = T_min[keep]
        n_out = int(M - keep.sum())

        # ── pausas usuales sobre el perfil mediano (sin outliers) ──
        t50_usual = np.percentile(T_usual, 50, axis=0)
        bumps: List[dict] = []
        if detect_bumps and len(T_usual) >= 20:
            bumps = _detect_bumps(
                t50_usual, x_km, pace_factor=bump_pace_factor,
                min_extra_min=bump_min_extra_min)
            for bp in bumps:
                _bump_trip_stats(bp, T_usual, x_km)
                geo = _geo_at_km(df_leg, bp["km"])
                bp["lat"], bp["lon"] = geo if geo else (None, None)
                bp.pop("i_from"), bp.pop("i_to")

        fig, axes = plt.subplots(5, 1, figsize=(12, 21), sharex=True)
        route_km = x_km[-1]
        rng = np.random.default_rng(seed)

        # A0 · sin filtros (crudo, directo de df_final)
        raw_ids = df_leg["trip_id"].unique().to_list()
        pick = set(rng.choice(len(raw_ids), size=min(n_background, len(raw_ids)),
                              replace=False).tolist())
        sample_ids = [raw_ids[i] for i in pick]
        for sub in (df_leg.filter(pl.col("trip_id").is_in(sample_ids))
                    .select(["trip_id", "d_km", "t_min"])
                    .partition_by("trip_id")):
            axes[0].plot(sub["d_km"].to_numpy(), sub["t_min"].to_numpy(),
                         color="#888888", lw=0.4, alpha=0.4)
        shown = (f"muestra de {len(sample_ids):,} de {n_raw:,}"
                 if len(sample_ids) < n_raw else f"{n_raw:,}")
        axes[0].set_title(
            f"A0 · Sin filtros — {shown} viajes válidos tal cual "
            f"(sin interpolar ni exigir cobertura)")

        # A · filtro leve
        _draw_band(axes[1], x_km, T_min, n_background, seed)
        axes[1].set_title(
            f"A · Filtro leve — {M:,} viajes "
            f"(cobertura ≥ {min_coverage:.0%} de {route_km:.1f} km)")

        # B · sin outliers
        _draw_band(axes[2], x_km, T_usual, n_background, seed)
        axes[2].set_title(
            f"B · Sin outliers — {len(T_usual):,} viajes "
            f"({n_out:,} excluidos; duración fuera de "
            f"[{max(lo, 0):.0f}, {hi:.0f}] min · Tukey {outlier_k}·IQR)")

        # C · clusters rápido/normal/lento sobre el perfil completo
        clusters_info: Dict[str, dict] = {}
        labels = None
        order = None
        if len(T_usual) >= 4 * k_clusters:
            labels, _ = _kmeans(T_usual, k_clusters, seed=seed)
            # ordenar clusters por duración mediana → rápido < normal < lento
            med_by_cluster = [np.median(T_usual[labels == j, -1])
                              for j in range(k_clusters)]
            order = np.argsort(med_by_cluster)
            names = (["rápido", "normal", "lento"] if k_clusters == 3
                     else [f"C{r + 1}" for r in range(k_clusters)])
            palette = ([_CLUSTER_STYLE[n] for n in names] if k_clusters == 3
                       else [f"C{r}" for r in range(k_clusters)])
            for rank, j in enumerate(order):
                name, color = names[rank], palette[rank]
                Tc = T_usual[labels == j]
                bg = rng.choice(len(Tc), size=min(n_background // k_clusters,
                                                  len(Tc)), replace=False)
                for i in bg:
                    axes[3].plot(x_km, Tc[i], color=color, lw=0.4, alpha=0.28)
                med_min = float(np.median(Tc[:, -1]))
                axes[3].plot(x_km, np.percentile(Tc, 50, axis=0), color=color,
                             lw=2.6, zorder=5,
                             label=f"{name} — {len(Tc):,} viajes · "
                                   f"mediana {med_min:.0f} min")
                clusters_info[name] = {"n": int(len(Tc)),
                                       "median_min": round(med_min, 1)}
            axes[3].legend(loc="upper left", fontsize=9)
            axes[3].set_title(
                f"C · Clusters k={k_clusters} (k-means sobre el perfil "
                f"distancia–tiempo, sin outliers)")
        else:
            axes[3].set_title(
                f"C · Clusters — omitido (solo {len(T_usual)} viajes sin "
                f"outliers; se requieren ≥ {4 * k_clusters})")

        # D · clusters por FORMA (perfil normalizado por la duración total):
        # separa por DÓNDE se gasta el tiempo, no por cuánto.
        shape_info: Dict[str, dict] = {}
        labels_s = None
        order_s = None
        if len(T_usual) >= 4 * k_clusters:
            T_norm = T_usual / T_usual[:, -1:]          # 0 → 1 del propio viaje
            labels_s, _ = _kmeans(T_norm, k_clusters, seed=seed)
            # fracción del tiempo consumida en la 1ª mitad de la ruta
            i_mid = T_norm.shape[1] // 2
            frac_mid = [np.median(T_norm[labels_s == j, i_mid])
                        for j in range(k_clusters)]
            order_s = np.argsort(frac_mid)[::-1]        # mayor → lento al inicio
            names_s = (list(_SHAPE_STYLE) if k_clusters == 3
                       else [f"forma {r + 1}" for r in range(k_clusters)])
            palette_s = (list(_SHAPE_STYLE.values()) if k_clusters == 3
                         else [f"C{r}" for r in range(k_clusters)])
            for rank, j in enumerate(order_s):
                name, color = names_s[rank], palette_s[rank]
                Tn = T_norm[labels_s == j] * 100.0      # % del tiempo total
                bg = rng.choice(len(Tn), size=min(n_background // k_clusters,
                                                  len(Tn)), replace=False)
                for i in bg:
                    axes[4].plot(x_km, Tn[i], color=color, lw=0.4, alpha=0.28)
                f_mid = float(np.median(Tn[:, i_mid]))
                axes[4].plot(x_km, np.percentile(Tn, 50, axis=0), color=color,
                             lw=2.6, zorder=5,
                             label=f"{name} — {len(Tn):,} viajes · "
                                   f"{f_mid:.0f}% del tiempo en la 1ª mitad")
                shape_info[name] = {"n": int(len(Tn)),
                                    "pct_time_first_half": round(f_mid, 1)}
            # referencia: consumo perfectamente proporcional a la distancia
            axes[4].plot([0, route_km], [0, 100], color="black", lw=1.0,
                         ls="--", alpha=0.6, label="proporcional a la distancia",
                         zorder=4)
            axes[4].legend(loc="upper left", fontsize=9)
            axes[4].set_title(
                f"D · Clusters por forma k={k_clusters} (perfil normalizado "
                f"por la duración de cada viaje, sin outliers)")
            axes[4].set_ylim(0, 102)
        else:
            axes[4].set_title(
                f"D · Clusters por forma — omitido (solo {len(T_usual)} viajes "
                f"sin outliers; se requieren ≥ {4 * k_clusters})")

        # ── pausas: línea vertical en todos los paneles + etiqueta arriba ──
        for bp in bumps:
            for ax in axes:
                ax.axvline(bp["km"], color=_BUMP_COLOR, ls=(0, (4, 3)),
                           lw=1.4, alpha=0.85, zorder=6)
            axes[0].annotate(f"‖ pausa km {bp['km']:.1f}",
                             xy=(bp["km"], 0.985),
                             xycoords=("data", "axes fraction"),
                             rotation=90, va="top", ha="right",
                             fontsize=8.5, color=_BUMP_COLOR)

        for ax in (axes[0], axes[1], axes[2], axes[3]):
            ax.set_ylabel("Tiempo desde el inicio (min)")
            ax.set_ylim(bottom=0)
        axes[4].set_ylabel("% del tiempo total del viaje")
        for ax in axes:
            ax.set_xlim(0, route_km)
        axes[4].set_xlabel("Distancia recorrida (km)")
        fig.suptitle(f"Trayectorias distancia–tiempo — {group_col} {leg}",
                     fontsize=14, y=0.996)

        footer = "\n".join(
            f"‖ pausa usual: km {bp['km']:.1f} ({bp['from_km']:.1f}–"
            f"{bp['to_km']:.1f}) · +{bp['extra_min_median_trip']:.1f} min "
            f"(mediana por viaje) · {bp['pct_trips_pausing']:.0f}% de viajes "
            f"con >1 min extra · lat {bp['lat']}, lon {bp['lon']}"
            for bp in bumps)
        bottom = 0.008 + 0.011 * len(bumps)
        if footer:
            fig.text(0.012, 0.004, footer, fontsize=9, color="#555",
                     va="bottom", linespacing=1.6)
        fig.tight_layout(rect=(0, bottom, 1, 0.988))

        safe = re.sub(r"[^0-9A-Za-z_]+", "", str(leg).replace("->", "_"))
        png = out_dir / f"trajectories_leg{safe}_{suffix}.png"
        fig.savefig(png, dpi=150)

        # ── HTML interactivo (liviano: bandas + medianas + muestra) ──
        html_path = None
        if make_html:
            from ._traj_html import build_html_payload, write_trajectories_html
            payload = build_html_payload(
                leg=str(leg), suffix=str(suffix), x_km=x_km,
                T_min=T_min, T_usual=T_usual,
                labels=labels, order=order, clusters_info=clusters_info,
                labels_s=labels_s, order_s=order_s, shape_info=shape_info,
                bumps=bumps, lo=lo, hi=hi, n_raw=n_raw,
                cluster_style=_CLUSTER_STYLE, shape_style=_SHAPE_STYLE,
                seed=seed)
            html_path = out_dir / f"trajectories_leg{safe}_{suffix}.html"
            write_trajectories_html(payload, html_path)

        if verbose:
            extra = (" · clusters: " + ", ".join(
                f"{k_} {v['n']} ({v['median_min']:.0f} min)"
                for k_, v in clusters_info.items())) if clusters_info else ""
            extra_s = (" · forma: " + ", ".join(
                f"{k_} {v['n']} ({v['pct_time_first_half']:.0f}% 1ª mitad)"
                for k_, v in shape_info.items())) if shape_info else ""
            print(f"  ✔ {group_col}={leg}: {M:,} viajes, {n_out:,} outliers"
                  f"{extra}{extra_s}")
            for bp in bumps:
                print(f"    ⏸ pausa km {bp['km']:.1f} "
                      f"({bp['from_km']:.1f}–{bp['to_km']:.1f}): "
                      f"+{bp['extra_min_median_trip']:.1f} min mediana · "
                      f"{bp['pct_trips_pausing']:.0f}% de viajes pausan · "
                      f"lat {bp['lat']}, lon {bp['lon']}")
            print(f"    → {png}" + (f"\n    → {html_path}" if html_path else ""))

        results[str(leg)] = {"png": png, "html": html_path,
                             "n_trips": M, "n_outliers": n_out,
                             "clusters": clusters_info,
                             "shape_clusters": shape_info,
                             "bumps": bumps,
                             "fig": fig if show else None}
        if not show:
            plt.close(fig)
    return results
