**Table S5. The round trip does not depend on where it is cut.**

Each row re-runs trip construction on January–August 2024 with the route's cut point moved from the one used in this paper (*used*) to the point where vehicles spend most time at rest (*rest*), a single rule applied to every route; only the routes where that rule picks a different point are shown. *Moved* is the distance between the two points along the loop. The trip-construction code is the one of this paper; the runs predate the models.

| Route | Cut used / rest (km) | Moved (km) | Median round trip, used / rest (min) | Δ (min) | Coverage, used / rest (%) | Inferred, used / rest (%) | Quality, used / rest |
|---|---|---|---|---|---|---|---|
| 101 | 24.26 / 23.99 | 0.3 | 149.5 / 149.9 | +0.4 | 69.8 / 69.8 | 0.4 / 0.5 | review / review |
| 102 | 0.00 / 32.63 | 0.3 | 115.5 / 115.9 | +0.4 | 82.2 / 82.5 | 0.5 / 0.8 | ok / ok |
| 301 | 22.35 / 9.63 | 12.7 | 134.9 / 134.6 | -0.3 | 78.1 / 76.1 | 1.4 / 1.3 | ok / ok |
| 501 | 0.00 / 10.43 | 10.4 | 139.2 / 139.0 | -0.2 | 75.3 / 76.1 | 0.7 / 3.0 | ok / ok |
| 902 | 0.00 / 9.88 | 9.9 | 59.0 / 59.1 | +0.1 | 85.4 / 89.2 | 1.5 / 1.5 | ok / ok |
| 1002 | 32.63 / 32.23 | 0.4 | 128.5 / 128.5 | +0.0 | 71.7 / 72.2 | 2.1 / 1.9 | ok / ok |
| 1501 | 6.43 / 28.58 | 22.1 | 156.9 / 156.9 | +0.0 | 70.6 / 76.7 | 2.5 / 9.5 | ok / review |
| 1502 | 0.38 / 0.58 | 0.2 | 94.8 / 94.1 | -0.7 | 82.7 / 82.3 | 2.9 / 2.5 | ok / ok |
| 1601 | 22.30 / 9.33 | 11.8 | 82.8 / 82.2 | -0.6 | 87.7 / 86.5 | 2.1 / 1.9 | ok / ok |
| 1602 | 2.57 / 15.44 | 11.9 | 82.8 / 82.5 | -0.3 | 86.8 / 85.6 | 2.6 / 2.7 | ok / ok |
| 1702 | 32.83 / 11.48 | 11.7 | 110.2 / 110.2 | +0.0 | 78.8 / 82.3 | 3.2 / 3.2 | ok / ok |
| 1703 | 28.15 / 40.65 | 12.5 | 118.2 / 118.7 | +0.5 | 76.8 / 81.3 | 3.6 / 4.3 | ok / ok |
| 1901 | 0.00 / 50.00 | 0.3 | 163.4 / 163.9 | +0.5 | 72.0 / 72.0 | 4.2 / 4.1 | ok / ok |
| 1902 | 0.00 / 29.53 | 20.1 | 163.9 / 163.3 | -0.6 | 69.2 / 75.4 | 3.9 / 10.3 | ok / poor |
| 20000 | 12.05 / 20.91 | 8.9 | 78.1 / 78.1 | +0.0 | 85.4 / 91.0 | 3.6 / 3.3 | ok / ok |

*Notes.* Moving the cut point by up to 22.1 km changes the median round trip by at most 0.7 min (median 0.3). What the point does change is how much of the service ends up in complete round trips and where crossings are timed: at the rest point 1501 would be rated review, its inferred crossings rising from 2.5 to 9.5%; 1902 would be rated poor, its inferred crossings rising from 3.9 to 10.3% — vehicles park at the rest point and stop reporting, so the crossing falls inside a gap.

The same holds across four single-cut rules (km 0 of the polyline, main layover point, consensus, dispatch point) compared on the 27 routes of the public release: on the 20 routes where the rules' points lie at least 5 km apart (up to 24.8 km), the median round trip differs across rules by at most 1.1 min (median 0.2), while coverage differs by up to 9.1 points.
