**Table S2. Per-route data funnel (January–August 2024 analysis window).**
Raw AVL records and cleaning retention are computed over the full downloaded
range. A *service stretch* is a vehicle's continuous operation on the route;
*coverage* is the share of the route's records that fall inside valid round
trips; *round trips* run from one crossing of the route's cut point to the
next within a stretch; *with shortcut* are those containing a forward jump
along the route that the vehicle could not have driven, which stay out of
the models; *modeled* round trips reach the per-route modeling dataset;
**grid** round trips are those retained by the trajectory matrix and
therefore actually analyzed. *Cut quality* applies the pre-specified
thresholds (Table S4).

| Route | UNE no. | Raw records | Kept (%) | Fleet | Active (90%) | Loop (km) | Service stretches | Coverage (%) | Round trips | With shortcut (%) | Modeled | Grid | Grid kept (%) | Cut quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 101 | 102 | 2,305,496 | 71.8 | 279 | 26 | 50.1 | 6,676 | 69.9 | 7,870 | 10.2 | 7,065 | 6,671 | 94.4 | review |
| 102 | 101 | 2,171,260 | 78.4 | 215 | 22 | 32.9 | 3,802 | 82.2 | 12,260 | 0.5 | 12,205 | 11,485 | 94.1 | ok |
| 201 | 201 | 1,598,485 | 80.8 | 202 | 28 | 57.6 | 5,348 | 72.6 | 5,346 | 0.9 | 5,298 | 4,975 | 93.9 | ok |
| 202 | 202 | 1,892,681 | 82.2 | 208 | 19 | 57.2 | 6,351 | 72.3 | 6,481 | 0.4 | 6,456 | 6,067 | 94.0 | ok |
| 301 | 301 | 2,082,918 | 84.2 | 234 | 27 | 45.7 | 5,603 | 78.1 | 10,591 | 7.7 | 9,775 | 9,277 | 94.9 | ok |
| 302 | 302 | 1,929,030 | 86.5 | 232 | 29 | 47.6 | 5,643 | 70.2 | 8,771 | 10.9 | 7,813 | 7,375 | 94.4 | review |
| 401 | 401 | 3,848,531 | 86.4 | 295 | 38 | 49.0 | 13,079 | 74.5 | 18,120 | 1.6 | 17,831 | 16,760 | 94.0 | ok |
| 402 | 402 | 3,367,837 | 88.0 | 279 | 40 | 48.9 | 10,432 | 74.8 | 16,203 | 1.9 | 15,891 | 14,634 | 92.1 | ok |
| 501 | 501 | 1,060,584 | 89.6 | 215 | 47 | 38.9 | 2,346 | 75.3 | 5,340 | 0.8 | 5,296 | 5,009 | 94.6 | ok |
| 601 | 601 | 2,104,594 | 90.7 | 219 | 19 | 64.8 | 5,537 | 73.5 | 8,536 | 12.6 | 7,463 | 7,039 | 94.3 | review |
| 801 | 801 | 1,533,580 | 95.0 | 221 | 30 | 36.0 | 4,159 | 72.3 | 9,008 | 1.5 | 8,873 | 8,405 | 94.7 | ok |
| 901 | 901 | 865,050 | 93.6 | 38 | 11 | 39.0 | 3,067 | 77.8 | 5,135 | 10.1 | 4,617 | 4,325 | 93.7 | review |
| 902 | 902 | 732,007 | 95.9 | 157 | 19 | 22.4 | 1,745 | 85.4 | 10,619 | 0.2 | 10,601 | 9,718 | 91.7 | ok |
| 1001 | 1002 | 1,706,972 | 98.1 | 241 | 25 | 42.4 | 8,193 | 75.2 | 10,671 | 6.1 | 10,024 | 9,402 | 93.8 | ok |
| 1002 | 1001 | 1,814,607 | 98.0 | 248 | 34 | 43.3 | 9,601 | 71.7 | 10,669 | 7.4 | 9,879 | 9,151 | 92.6 | ok |
| 1101 | 1101 | 1,208,707 | 99.9 | 130 | 15 | 47.3 | 2,786 | 72.1 | 5,357 | 0.4 | 5,334 | 4,999 | 93.7 | ok |
| 1101-ALT (110101) | 1101-ALT | 1,071,408 | 99.3 | 158 | 15 | 47.3 | 2,537 | 67.4 | 4,406 | 0.5 | 4,385 | 4,110 | 93.7 | ok |
| 1201 | 1201 | 1,631,661 | 98.6 | 226 | 19 | 49.0 | 4,184 | 75.2 | 7,939 | 0.7 | 7,883 | 7,379 | 93.6 | ok |
| 1301 | 1301 | 1,029,024 | 98.2 | 199 | 28 | 38.7 | 2,551 | 78.9 | 6,202 | 0.8 | 6,155 | 5,826 | 94.7 | ok |
| 1401 | 1401 | 945,553 | 99.2 | 204 | 32 | 43.1 | 2,276 | 74.1 | 5,303 | 3.1 | 5,136 | 4,851 | 94.5 | ok |
| 1402 | 1402 | 929,036 | 99.4 | 201 | 33 | 42.9 | 2,401 | 73.6 | 5,077 | 3.0 | 4,926 | 4,611 | 93.6 | ok |
| 1501 | 1501 | 2,156,635 | 98.8 | 253 | 23 | 45.7 | 5,601 | 70.6 | 10,612 | 0.9 | 10,512 | 9,954 | 94.7 | ok |
| 1502 | 1502 | 1,486,067 | 98.9 | 190 | 15 | 25.4 | 4,375 | 82.7 | 14,219 | 3.4 | 13,741 | 12,758 | 92.8 | ok |
| 1601 | 1602 | 756,546 | 98.7 | 108 | 4 | 24.8 | 1,522 | 87.7 | 8,465 | 0.3 | 8,437 | 7,941 | 94.1 | ok |
| 1602 | 1601 | 711,383 | 98.5 | 164 | 23 | 24.8 | 1,631 | 86.8 | 7,941 | 0.6 | 7,897 | 7,463 | 94.5 | ok |
| 1701 | 1701 | 2,341,479 | 98.5 | 223 | 15 | 40.0 | 8,645 | 80.9 | 15,170 | 0.5 | 15,088 | 14,254 | 94.5 | ok |
| 1702 | 1702 | 2,011,258 | 98.3 | 245 | 18 | 33.0 | 5,982 | 78.8 | 15,358 | 0.1 | 15,338 | 14,457 | 94.3 | ok |
| 1703 | 1703 | 3,475,405 | 96.9 | 300 | 26 | 40.7 | 9,945 | 76.8 | 24,435 | 1.5 | 24,072 | 22,742 | 94.5 | ok |
| 1801 | 1801 | 930,184 | 99.0 | 200 | 39 | 44.1 | 3,140 | 78.1 | 5,699 | 0.8 | 5,654 | 5,292 | 93.6 | ok |
| 1801-ALT (180101) | 1801-ALT | 775,878 | 99.7 | 150 | 22 | 43.5 | 2,620 | 80.2 | 4,956 | 0.8 | 4,917 | 4,644 | 94.4 | ok |
| 1802 | 1802 | 823,398 | 98.8 | 152 | 27 | 45.2 | 2,344 | 78.4 | 5,059 | 0.7 | 5,023 | 4,727 | 94.1 | ok |
| 1802-ALT (180201) | 1802-ALT | 787,073 | 99.8 | 125 | 21 | 44.6 | 2,533 | 79.7 | 4,994 | 0.7 | 4,958 | 4,660 | 94.0 | ok |
| 1901 | 1901 | 2,009,360 | 98.0 | 242 | 19 | 50.3 | 7,913 | 72.0 | 9,568 | 1.5 | 9,425 | 8,943 | 94.9 | ok |
| 1902 | 1902 | 1,540,778 | 96.5 | 197 | 17 | 49.6 | 5,838 | 69.2 | 6,959 | 1.8 | 6,831 | 6,467 | 94.7 | ok |
| 20000 | 2201 | 394,211 | 98.9 | 61 | 3 | 25.0 | 820 | 85.4 | 4,644 | 0.5 | 4,622 | 4,368 | 94.5 | ok |
| **Total (35)** |  | **56,028,676** |  |  |  |  | **171,226** |  | **317,983** |  | **309,421** | **290,739** | **94.0** |  |

*Notes.* Route numbers are those of the AVL records; *UNE no.* is the
number under which the operator's current catalogue lists the route, which
differs for 7 of them. The variant routes are the two services of four
route codes that the AVL records do not distinguish, separated vehicle by
vehicle (Table S1). *Fleet* is the number of distinct vehicles observed on
the route; *Active (90%)* is the number of vehicles that perform 90% of the
route's modeled round trips — a median of 14.2% of the vehicles that operate a route.
Stretches end at the end of the service day (90,559),
at a gap of more than 20 minutes between records (40,649)
or at a jump along the route that cannot have been driven (40,018); only round trips completed within one stretch
are kept, which is why coverage falls short of 100% (median 75.2%, range 67.4–87.7%).
119 round trips that travelled more than 1.5 times
the loop (a missed crossing) are not counted as valid, and 8,562 valid round trips with a shortcut stay out of
the models: the grid cells they skip have no observed time.
4 routes are rated *review* (101, 302, 601, 901) and kept under the pre-specified criterion; none is rated *poor*.

*Grid* is the last stage of the funnel and the one that defines what
is actually analyzed: the trajectory matrix places a common distance grid,
laid on the progress along the route polyline from the cut point, at the
5th percentile of round-trip length and keeps the round trips
that cover it completely, truncating the longer ones, which discards distance
beyond the grid but no round trips. It
retains 94.0% of modeled round trips overall (94.1% median across the 35 routes). Grid round trips are the denominator of every
model comparison in this paper.

For tractability the cross-validation is capped at 2,500 round trips per route (a uniform random subsample applied identically to every algorithm, so the comparison remains paired); 35 of the 35 routes hit the cap, leaving 87,500 round trips in the cross-validation. The cap bounds the cost of the nearest-neighbour and kernel baselines, which are quadratic in the number of round trips.
