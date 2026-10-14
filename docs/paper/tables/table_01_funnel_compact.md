**Table 1. Data funnel (January–August 2024 analysis window).**

Routes are grouped by where their AVL records come from; the per-route
funnel is Table S2 in the Supplementary Information. Raw AVL records and cleaning
retention are computed over the full downloaded range. A *service stretch* is a
vehicle's continuous operation on the route; *coverage* is the share of a route's
records that fall inside valid round trips; a *round trip* runs from one crossing of
the route's cut point to the next within a stretch; *modeled* round trips exclude
those with a shortcut and reach the per-route modeling dataset; **grid** round trips
are those retained by the trajectory matrix and therefore actually analyzed, and are
the denominator of every model comparison in this paper. Counts are sums over the
routes of each group; *kept* is weighted by raw records, *grid kept* is recomputed
from the sums, and *coverage* and *median round trip* are medians of the per-route
values. The variant routes are the two services of four route codes that the AVL
records do not distinguish, separated vehicle by vehicle (Table S1).

| Group | Routes | Raw records | Kept (%) | Service stretches | Coverage (%), median | Round trips | Modeled | Grid | Grid kept (%) | Median round trip (min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Routes in the public AVL release | 27 | 48,557,439 | 91.4 | 150,589 | 75.2 | 277,132 | 269,088 | 252,845 | 94.0 | 134.9 |
| Variant routes, separated from four shared route codes | 8 | 7,471,237 | 99.4 | 20,637 | 76.1 | 40,851 | 40,333 | 37,894 | 94.0 | 141.0 |
| **All routes** | 35 | 56,028,676 | 92.4 | 171,226 | 75.2 | 317,983 | 309,421 | 290,739 | 94.0 | 138.0 |
