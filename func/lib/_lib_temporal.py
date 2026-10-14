"""
_lib_temporal.py
================
Análisis y filtro de consistencia temporal.

Este archivo es parte interna del pipeline (módulo `_lib_*`); los pasos
`step_*.py` lo importan, pero NO debe usarse directamente desde notebooks.

Origen
------
Copia exacta de `temporal_consistency.py` del proyecto BUSONORA original. Mantenerlo
sincronizado si la versión upstream cambia.
"""

#################
### Libraries ###
#################
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional, Tuple, Union
from pathlib import Path


#──────────────────────────────────────────#
# ── 1. Detección de huecos temporales ── #
#──────────────────────────────────────────#

def analyze_temporal_gaps(
    df_rec: pl.DataFrame,
    time_col: str = "time",
    bus_col: str = "bus_code",
    expected_gap_min: int = 1,
    min_gap_min: int = 2,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Detecta huecos temporales en los registros GPS por camión.

    Para cada unidad ordena los registros por tiempo, calcula la diferencia
    entre registros consecutivos en minutos y marca como hueco toda diferencia
    mayor o igual a ``min_gap_min``.

    Parameters
    ----------
        df_rec (pl.DataFrame):   DataFrame con registros GPS.
        time_col (str):          Columna de timestamp. Default: ``"time"``
        bus_col (str):           Columna de identificador del camión.
                                 Default: ``"bus_code"``
        expected_gap_min (int):  Intervalo esperado entre registros (min).
                                 Default: ``1``
        min_gap_min (int):       Diferencia mínima para considerar un hueco.
                                 Default: ``2``
        verbose (bool):          Imprime estadísticas básicas. Default: False

    Returns
    -------
        df_gaps (pl.DataFrame): Un registro por hueco con columnas:
            - bus_col       : identificador del camión
            - ``gap_start`` : timestamp donde inicia el hueco
            - ``gap_end``   : timestamp donde termina el hueco
            - ``gap_min``   : duración del hueco en minutos
            - ``lost_min``  : minutos perdidos = gap_min - expected_gap_min

    Examples
    --------
        >>> df_gaps = analyze_temporal_gaps(df_rec_101, verbose=True)
    """
    if verbose:
        print(f"[temporal_consistency] {df_rec.shape[0]:,} registros | "
              f"{df_rec[bus_col].n_unique()} camiones")

    if df_rec[time_col].dtype == pl.Utf8:
        df_rec = df_rec.with_columns(
            pl.col(time_col).str.to_datetime().alias(time_col)
        )

    df_gaps = (
        df_rec
        .select([bus_col, time_col])
        .sort([bus_col, time_col])
        .with_columns(
            pl.col(time_col).shift(-1).over(bus_col).alias("_next_time")
        )
        .filter(pl.col("_next_time").is_not_null())
        .with_columns(
            (pl.col("_next_time") - pl.col(time_col))
            .dt.total_minutes().cast(pl.Int64).alias("_diff_min")
        )
        .filter(pl.col("_diff_min") >= min_gap_min)
        .select([
            pl.col(bus_col),
            pl.col(time_col).alias("gap_start"),
            pl.col("_next_time").alias("gap_end"),
            pl.col("_diff_min").alias("gap_min"),
            (pl.col("_diff_min") - expected_gap_min).alias("lost_min"),
        ])
    )

    if verbose:
        print(f"[temporal_consistency] Huecos detectados : {df_gaps.shape[0]:,}")
        print(f"[temporal_consistency] Minutos perdidos  : {df_gaps['lost_min'].sum():,}")
        print(f"[temporal_consistency] Duración promedio : {df_gaps['gap_min'].mean():.1f} min")

    return df_gaps


#──────────────────────────────────────────#
# ── 2. Distribución global por duración ── #
#──────────────────────────────────────────#

def report_gaps_global(
    df_gaps: pl.DataFrame,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Distribución global de huecos: cantidad de huecos por duración en minutos.

    Retorna todos los valores sin filtrar; el recorte del eje X ocurre en
    ``plot_gap_distribution`` mediante ``max_gap_display`` o ``pct_cutoff``.

    Parameters
    ----------
        df_gaps (pl.DataFrame):  Salida de ``analyze_temporal_gaps``.
        verbose (bool):          Imprime la tabla completa. Default: False

    Returns
    -------
        df_global (pl.DataFrame): Columnas:
            - ``gap_min``      : duración en minutos
            - ``n_gaps``       : número de huecos con esa duración
            - ``pct``          : porcentaje sobre el total de huecos
            - ``cum_lost_min`` : minutos perdidos acumulados hasta esa duración

    Examples
    --------
        >>> df_global = report_gaps_global(df_gaps)
    """
    df_global = (
        df_gaps
        .group_by("gap_min")
        .agg([
            pl.len().alias("n_gaps"),
            pl.col("lost_min").sum().alias("total_lost_min"),
        ])
        .sort("gap_min")
        .with_columns([
            (pl.col("n_gaps") / pl.col("n_gaps").sum() * 100).round(2).alias("pct"),
            pl.col("total_lost_min").cum_sum().alias("cum_lost_min"),
        ])
    )

    if verbose:
        print("[temporal_consistency] ── Distribución global ──")
        print(df_global)

    return df_global


#──────────────────────────────────────────────────────#
# ── 3. Gráfico: distribución global de huecos ── #
#──────────────────────────────────────────────────────#

def plot_gap_distribution(
    df_global: pl.DataFrame,
    max_gap_display: Optional[int] = None,
    pct_cutoff: float = 99.0,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
    n_total_records: Optional[int] = None,
    verbose: bool = False,
) -> plt.Figure:
    """
    Gráfico de un panel: barras de cantidad de huecos por duración con una
    línea secundaria que muestra el porcentaje acumulado del total de huecos.

    El eje X se recorta automáticamente al punto donde se alcanza ``pct_cutoff``
    (default 99%) salvo que se especifique ``max_gap_display`` manualmente.

    Si se pasa ``n_total_records``, el subtítulo muestra qué porcentaje del
    total de registros GPS representan los registros adyacentes a huecos
    (cada hueco elimina 2 registros: gap_start y gap_end).

    Parameters
    ----------
        df_global (pl.DataFrame): Salida de ``report_gaps_global``.
        max_gap_display (int):    Duración máxima a mostrar en eje X (minutos).
                                  Si es None se calcula automáticamente usando
                                  ``pct_cutoff``. Default: None
        pct_cutoff (float):       Percentil acumulado para el corte automático
                                  del eje X. Solo aplica cuando ``max_gap_display``
                                  es None. Default: ``99.0``
        figsize (tuple):          Tamaño de figura. Default: ``(10, 5)``
        save_path (str):          Ruta para guardar la figura. Default: None
        n_total_records (int):    Número total de registros GPS en df_rec. Si se
                                  pasa, se anota el % de registros adyacentes
                                  a huecos en el subtítulo. Default: None
        verbose (bool):           Imprime el corte aplicado. Default: False

    Returns
    -------
        fig (plt.Figure): Figura de matplotlib.

    Examples
    --------
        >>> # Corte automático al 99%
        >>> fig = plot_gap_distribution(df_global)

        >>> # Con contexto de % sobre registros totales
        >>> fig = plot_gap_distribution(df_global, n_total_records=df_rec.shape[0])
    """
    if verbose:
        print("[temporal_consistency] Generando gráfico de distribución global...")

    # ── Calcular % acumulado sobre el total de huecos ──
    total_gaps = df_global["n_gaps"].sum()
    df_plot = (
        df_global
        .sort("gap_min")
        .with_columns(
            (pl.col("n_gaps").cum_sum() / total_gaps * 100)
            .round(2).alias("cum_pct")
        )
    )

    # ── Determinar corte del eje X ──
    if max_gap_display is None:
        cutoff_row = df_plot.filter(pl.col("cum_pct") >= pct_cutoff).head(1)
        if cutoff_row.is_empty():
            max_gap_display = int(df_plot["gap_min"].max())
        else:
            max_gap_display = int(cutoff_row["gap_min"][0])
        if verbose:
            print(f"[temporal_consistency] Corte automático ({pct_cutoff}%): "
                  f"{max_gap_display} min")
    else:
        if verbose:
            print(f"[temporal_consistency] Corte manual: {max_gap_display} min")

    df_plot = df_plot.filter(pl.col("gap_min") <= max_gap_display)
    x       = df_plot["gap_min"].to_list()
    n_gaps  = df_plot["n_gaps"].to_list()
    cum_pct = df_plot["cum_pct"].to_list()

    # ── Figura ──
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Barras
    bars = ax1.bar(x, n_gaps, color="#E07B54", edgecolor="white",
                   linewidth=0.5, zorder=2)
    ax1.set_xlabel("Duración del hueco (minutos)", fontsize=11)
    ax1.set_ylabel("Número de huecos", fontsize=11, color="#E07B54")
    ax1.tick_params(axis='y', labelcolor="#E07B54")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax1.set_xticks(x)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(False)

    # Escala: ax1 va 0→y_max, ax2 va 0→100
    # Para que las líneas grises coincidan con los ticks de ax2,
    # cada 10% en ax2 equivale a (10/100)*y_max en ax1
    y_max = max(n_gaps) * 1.12
    ax1.set_ylim(0, y_max)

    for pct_line in range(0, 101, 10):
        y_in_ax1 = pct_line / 100 * y_max
        ax1.axhline(y_in_ax1, color="grey", linewidth=0.8,
                    linestyle=(0, (5, 8)), alpha=0.45, zorder=1)

    # Anotaciones en barras grandes
    for bar, val in zip(bars, n_gaps):
        if val >= max(n_gaps) * 0.05:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(n_gaps) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=7
            )

    # Línea acumulada — ax2 va estrictamente de 0 a 100
    ax2.plot(x, cum_pct, color="#3A7DC9", linewidth=2.2,
             marker="o", markersize=3.5, zorder=3)
    ax2.set_ylabel("% acumulado de huecos", fontsize=11, color="#3A7DC9")
    ax2.tick_params(axis='y', labelcolor="#3A7DC9")
    ax2.set_ylim(0, 100)
    ax2.set_yticks(range(0, 101, 10))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}%"))

    # Línea de referencia pct_cutoff
    ax2.axhline(pct_cutoff, color="#3A7DC9", linewidth=1,
                linestyle="--", alpha=0.5)
    ax2.text(x[-1], pct_cutoff, f"{pct_cutoff:.0f}%",
             color="#3A7DC9", fontsize=8, ha="right", va="bottom")

    fig.suptitle("Distribución global de huecos temporales",
                 fontsize=13, fontweight="bold")

    # ── Subtítulo con contexto de impacto sobre el total de registros ──
    if n_total_records is not None and n_total_records > 0:
        total_gaps_all = int(df_global["n_gaps"].sum())
        # Cada hueco elimina como máximo 2 registros (gap_start, gap_end);
        # se acota a n_total_records porque puede haber solapamientos.
        approx_records_at_gap_edges = min(2 * total_gaps_all, n_total_records)
        pct_records = approx_records_at_gap_edges / n_total_records * 100
        subtitle = (
            f"n = {total_gaps_all:,} huecos  |  "
            f"≈ {approx_records_at_gap_edges:,} registros adyacentes  "
            f"({pct_records:.2f}% de {n_total_records:,} registros GPS totales)"
        )
        fig.text(0.5, 0.93, subtitle, ha="center", va="top",
                 fontsize=10, color="#444")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if verbose:
            print(f"[temporal_consistency] Guardado en: {save_path}")

    return fig


#──────────────────────────────────────────────────────#
# ── 4. Clasificación de calidad por camión ── #
#──────────────────────────────────────────────────────#

def make_data_quality(
    df_rec: pl.DataFrame,
    df_gaps: pl.DataFrame,
    bus_col: str = "bus_code",
    time_col: str = "time",
    long_gap_min: int = 10,
    thr_alta: float = 0.85,
    thr_media: float = 0.65,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Clasifica cada camión en alta / media / baja calidad de datos.

    Calcula dos métricas por camión a partir de ``df_gaps``:

    - **cobertura** : fracción del tiempo activo con datos presentes.
      ``(total_min_activo - lost_min) / total_min_activo``. Mayor es mejor.
    - **frag_rate** : proporción de huecos largos (> ``long_gap_min``) sobre
      el total. Mayor frag_rate → más interrupciones severas. Menor es mejor.

    Score final (0–1):
        ``score = 0.6 * cobertura + 0.4 * (1 - frag_rate)``

    Parameters
    ----------
        df_rec (pl.DataFrame):   DataFrame original de registros GPS.
        df_gaps (pl.DataFrame):  Salida de ``analyze_temporal_gaps``.
        bus_col (str):           Columna identificadora del camión.
                                 Default: ``"bus_code"``
        time_col (str):          Columna de timestamp. Default: ``"time"``
        long_gap_min (int):      Umbral para hueco "largo" (min). Default: ``10``
        thr_alta (float):        Score mínimo para calidad alta. Default: ``0.85``
        thr_media (float):       Score mínimo para calidad media. Default: ``0.65``
        verbose (bool):          Imprime distribución de calidad. Default: False

    Returns
    -------
        df_quality (pl.DataFrame): Una fila por camión con columnas:
            - bus_col         : identificador del camión
            - ``cobertura``   : fracción del tiempo cubierta (0–1)
            - ``frag_rate``   : proporción de huecos largos (0–1)
            - ``score``       : puntuación final (0–1)
            - ``data_quality``: ``"alta"``, ``"media"`` o ``"baja"``

    Examples
    --------
        >>> df_quality = make_data_quality(df_rec_101, df_gaps, verbose=True)
    """
    # Tiempo activo total por camión
    df_active = (
        df_rec.group_by(bus_col)
        .agg([
            pl.col(time_col).min().alias("t_min"),
            pl.col(time_col).max().alias("t_max"),
        ])
        .with_columns(
            ((pl.col("t_max") - pl.col("t_min")).dt.total_minutes())
            .cast(pl.Int64).alias("total_min_activo")
        )
    )

    # Métricas de huecos por camión
    df_bus = (
        df_gaps.group_by(bus_col)
        .agg([
            pl.len().alias("n_gaps"),
            pl.col("lost_min").sum().alias("total_lost_min"),
            (pl.col("gap_min") > long_gap_min).sum().alias("n_long_gaps"),
        ])
        .with_columns(
            (pl.col("n_long_gaps") / pl.col("n_gaps")).alias("frag_rate")
        )
    )

    df_quality = (
        df_bus.join(df_active, on=bus_col, how="left")
        .with_columns(
            (1.0 - pl.col("total_lost_min") / pl.col("total_min_activo"))
            .clip(0.0, 1.0).alias("cobertura")
        )
        .with_columns(
            (0.6 * pl.col("cobertura") + 0.4 * (1.0 - pl.col("frag_rate")))
            .round(4).alias("score")
        )
        .with_columns(
            pl.when(pl.col("score") >= thr_alta).then(pl.lit("alta"))
            .when(pl.col("score") >= thr_media).then(pl.lit("media"))
            .otherwise(pl.lit("baja"))
            .alias("data_quality")
        )
        .select([bus_col, "cobertura", "frag_rate", "score", "data_quality"])
        .sort("score", descending=True)
    )

    if verbose:
        dist = df_quality.group_by("data_quality").agg(pl.len().alias("n")).sort("data_quality")
        total = df_quality.shape[0]
        print(f"\n[temporal_consistency] ── Calidad de datos ──")
        for row in dist.iter_rows(named=True):
            pct = row["n"] / total * 100
            bar = "█" * int(pct / 2)
            print(f"  {row['data_quality']:6s}: {row['n']:>4d} camiones ({pct:5.1f}%)  {bar}")

    return df_quality


#──────────────────────────────────────────────────────#
# ── 5. Filtro temporal por umbral de hueco ── #
#──────────────────────────────────────────────────────#

def apply_temporal_filter(
    df_rec: pl.DataFrame,
    df_gaps: pl.DataFrame,
    gap_filter: Union[int, float, None],
    time_col: str = "time",
    bus_col: str = "bus_code",
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Filtra ``df_rec`` removiendo los registros adyacentes a huecos cuya
    duración excede un umbral. Un hueco grande rompe la continuidad de la
    trayectoria, por lo que tanto el último registro antes del hueco como
    el primero después se eliminan (no se pueden interpretar como parte
    de un movimiento continuo).

    Interpretación de ``gap_filter``:
        - **Entero** (ej. ``3``):  umbral absoluto en minutos. Se eliminan
          los registros adyacentes a huecos con ``gap_min >= 3``.
        - **Flotante** (ej. ``0.9``): percentil (90%) sobre la distribución
          empírica de duraciones de huecos. Se calcula el umbral como el
          P90 de ``gap_min`` y se aplica como en el caso entero.
        - **None**: sin filtro; se devuelve ``df_rec`` sin modificar.

    Parameters
    ----------
        df_rec (pl.DataFrame):   DataFrame original de registros GPS.
        df_gaps (pl.DataFrame):  Salida de ``analyze_temporal_gaps``.
        gap_filter (int|float):  Umbral de hueco — entero=minutos, decimal=percentil.
        time_col (str):          Columna de timestamp. Default: ``"time"``
        bus_col (str):           Columna del camión. Default: ``"bus_code"``
        verbose (bool):          Imprime efecto del filtro. Default: False

    Returns
    -------
        df_filter_temporal (pl.DataFrame): df_rec sin los registros adyacentes
            a huecos por encima del umbral.

    Examples
    --------
        >>> # Filtrar por umbral absoluto de 3 minutos
        >>> df_filt = apply_temporal_filter(df_rec, df_gaps, gap_filter=3)

        >>> # Filtrar por percentil 90 (los huecos más largos del 10%)
        >>> df_filt = apply_temporal_filter(df_rec, df_gaps, gap_filter=0.9)
    """
    # ── Determinar umbral en minutos según tipo de gap_filter ──
    if gap_filter is None:
        # None = sin filtro: devolver df_rec sin modificar
        if verbose:
            print("[temporal_consistency] gap_filter=None → sin filtro temporal, "
                  "se devuelve df_rec sin cambios.")
        return df_rec

    is_percentile = isinstance(gap_filter, float) and (0.0 < gap_filter < 1.0)

    if is_percentile:
        if df_gaps.is_empty():
            threshold_min = 0
        else:
            threshold_min = int(
                df_gaps["gap_min"].quantile(gap_filter, "linear") or 0
            )
        mode_str = f"percentil {gap_filter*100:.0f}% → {threshold_min} min"
    else:
        threshold_min = int(gap_filter)
        mode_str = f"umbral absoluto {threshold_min} min"

    if verbose:
        print(f"\n[temporal_consistency] ── Filtro temporal ──")
        print(f"[temporal_consistency] Modo: {mode_str}")

    # ── Identificar huecos por encima del umbral ──
    df_bad_gaps = df_gaps.filter(pl.col("gap_min") >= threshold_min)
    n_bad_gaps  = df_bad_gaps.shape[0]

    if n_bad_gaps == 0:
        if verbose:
            print(f"[temporal_consistency] No hay huecos >= {threshold_min} min, "
                  f"se devuelve df_rec sin cambios.")
        return df_rec

    # ── Construir set de timestamps a eliminar (gap_start y gap_end por bus) ──
    # Cada hueco "malo" → eliminar 2 registros: el de gap_start y el de gap_end
    df_to_remove = pl.concat([
        df_bad_gaps.select([
            pl.col(bus_col),
            pl.col("gap_start").alias(time_col),
        ]),
        df_bad_gaps.select([
            pl.col(bus_col),
            pl.col("gap_end").alias(time_col),
        ]),
    ]).unique()

    # ── Anti-join: dejar solo los que NO están en df_to_remove ──
    df_filter_temporal = df_rec.join(
        df_to_remove,
        on=[bus_col, time_col],
        how="anti",
    )

    if verbose:
        n_in     = df_rec.shape[0]
        n_out    = df_filter_temporal.shape[0]
        n_bus_in = df_rec[bus_col].n_unique()
        n_bus_out = df_filter_temporal[bus_col].n_unique()
        n_removed = n_in - n_out
        pct_removed = n_removed / n_in * 100 if n_in > 0 else 0.0
        print(f"[temporal_consistency] Huecos >= {threshold_min} min : {n_bad_gaps:,}")
        print(f"[temporal_consistency] Registros eliminados        : "
              f"{n_removed:,} ({pct_removed:.2f}%)")
        print(f"[temporal_consistency] Registros antes / después   : "
              f"{n_in:,} → {n_out:,}")
        print(f"[temporal_consistency] Camiones antes / después    : "
              f"{n_bus_in} → {n_bus_out}")

    return df_filter_temporal


#──────────────────────────────────────────#
# ── 6. Pipeline completo ── #
#──────────────────────────────────────────#

def run_temporal_consistency(
    df_rec: pl.DataFrame,
    time_col: str = "time",
    bus_col: str = "bus_code",
    expected_gap_min: int = 1,
    min_gap_min: int = 2,
    max_gap_display: Optional[int] = None,
    pct_cutoff: float = 99.0,
    long_gap_min: int = 10,
    thr_alta: float = 0.85,
    thr_media: float = 0.65,
    gap_filter: Union[int, float] = 3,
    show_plot: bool = True,
    save_path: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Pipeline completo: detecta huecos, calcula distribución, genera gráfico,
    clasifica camiones por calidad y aplica un filtro temporal por umbral.

    Parameters
    ----------
        df_rec (pl.DataFrame):   DataFrame con registros GPS.
        time_col (str):          Columna de timestamp. Default: ``"time"``
        bus_col (str):           Columna de identificador del camión.
                                 Default: ``"bus_code"``
        expected_gap_min (int):  Intervalo esperado entre registros (min).
                                 Default: ``1``
        min_gap_min (int):       Umbral mínimo para considerar un hueco (min).
                                 Default: ``2``
        max_gap_display (int):   Duración máxima en eje X (minutos). Si es None
                                 se calcula automáticamente con ``pct_cutoff``.
                                 Default: None
        pct_cutoff (float):      Percentil acumulado para corte automático del
                                 eje X. Solo aplica si ``max_gap_display`` es None.
                                 Default: ``99.0``
        long_gap_min (int):      Umbral para hueco "largo" en calidad. Default: ``10``
        thr_alta (float):        Score mínimo para calidad alta. Default: ``0.85``
        thr_media (float):       Score mínimo para calidad media. Default: ``0.65``
        gap_filter (int|float):  Umbral del filtro temporal aplicado en df_rec:
                                 - entero (ej. ``3``)   → minutos absolutos
                                 - decimal (ej. ``0.9``) → percentil (P90)
                                 Default: ``3``
        show_plot (bool):        Mostrar figura con plt.show(). Default: True
        save_path (str):         Ruta para guardar el gráfico. Default: None
        verbose (bool):          Modo detallado. Default: False

    Returns
    -------
        df_gaps (pl.DataFrame):            Un registro por hueco detectado.
        df_global (pl.DataFrame):          Distribución global por duración.
        df_quality (pl.DataFrame):         Calidad por camión (score + etiqueta).
        df_filter_temporal (pl.DataFrame): df_rec filtrado según ``gap_filter``.

    Examples
    --------
        >>> df_gaps, df_global, df_quality, df_filter = run_temporal_consistency(
        ...     df_rec=df_rec_101,
        ...     gap_filter=3,            # 3 minutos
        ...     save_path="../data/02_interim/reports/101/gap_distribution.png",
        ...     verbose=True,
        ... )
        >>>
        >>> # Filtrar por percentil 90
        >>> df_gaps, df_global, df_quality, df_filter = run_temporal_consistency(
        ...     df_rec=df_rec_101,
        ...     gap_filter=0.9,
        ...     verbose=True,
        ... )
    """
    #──────────────────────────────────────────#
    # ── 6.1 Detectar huecos ── #
    #──────────────────────────────────────────#
    df_gaps = analyze_temporal_gaps(
        df_rec=df_rec,
        time_col=time_col,
        bus_col=bus_col,
        expected_gap_min=expected_gap_min,
        min_gap_min=min_gap_min,
        verbose=verbose,
    )

    #──────────────────────────────────────────#
    # ── 6.2 Distribución global ── #
    #──────────────────────────────────────────#
    df_global = report_gaps_global(
        df_gaps=df_gaps,
        verbose=verbose,
    )

    #──────────────────────────────────────────#
    # ── 6.3 Gráfico ── #
    #──────────────────────────────────────────#
    plot_gap_distribution(
        df_global=df_global,
        max_gap_display=max_gap_display,
        pct_cutoff=pct_cutoff,
        save_path=save_path,
        n_total_records=df_rec.shape[0],
        verbose=verbose,
    )

    if show_plot:
        plt.show()

    #──────────────────────────────────────────#
    # ── 6.4 Calidad por camión ── #
    #──────────────────────────────────────────#
    df_quality = make_data_quality(
        df_rec=df_rec,
        df_gaps=df_gaps,
        bus_col=bus_col,
        time_col=time_col,
        long_gap_min=long_gap_min,
        thr_alta=thr_alta,
        thr_media=thr_media,
        verbose=verbose,
    )

    #──────────────────────────────────────────#
    # ── 6.5 Filtro temporal sobre df_rec ── #
    #──────────────────────────────────────────#
    df_filter_temporal = apply_temporal_filter(
        df_rec=df_rec,
        df_gaps=df_gaps,
        gap_filter=gap_filter,
        time_col=time_col,
        bus_col=bus_col,
        verbose=verbose,
    )

    return df_gaps, df_global, df_quality, df_filter_temporal