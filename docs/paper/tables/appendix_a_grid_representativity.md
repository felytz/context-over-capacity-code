**Appendix A. What the grid placement retains, and from whom.**

The trajectory matrix places a common distance grid, laid on the progress along the route polyline from the cut point, at the `q`-quantile of round-trip length within each route and keeps round trips covering at least `min_coverage` of it. The table sweeps both parameters over the 35 routes. Standardized mean differences compare the **retained sample against the route's full population** of round trips — not against the discarded ones, which is the contrast that answers whether the analysed sample represents the route. `|SMD| > 0.25` is the conventional threshold for a material shift. The last column is the share of the route's vehicles that appear at all in the retained sample. The row in bold is the regime used in this paper (q = 0.05, coverage 1.0). A dash marks a regime that retains too few round trips to compare.

| q | min. coverage | routes | median retention | \|SMD\| length | \|SMD\| duration | \|SMD\| departure hour | TVD month | routes with \|SMD\| > 0.25 | vehicles retained |
|---|---|---|---|---|---|---|---|---|---|
| 0.95 | 0.99 | 35 | 77.0% | 0.264 | 0.065 | 0.020 | 0.020 | 20/35 | 93.8% |
| 0.50 | 0.99 | 35 | 90.3% | 0.198 | 0.028 | 0.011 | 0.013 | 6/35 | 98.5% |
| 0.10 | 0.99 | 35 | 96.8% | 0.118 | 0.015 | 0.008 | 0.006 | 0/35 | 99.5% |
| 0.05 | 0.99 | 35 | 97.7% | 0.106 | 0.013 | 0.005 | 0.005 | 0/35 | 100.0% |
| 0.95 | 1.00 | 35 | 0.1% | 0.779 | 0.238 | 0.141 | 0.144 | 18/35 | 16.3% |
| 0.50 | 1.00 | 35 | 37.5% | 0.529 | 0.118 | 0.051 | 0.032 | 35/35 | 76.9% |
| 0.10 | 1.00 | 35 | 87.7% | 0.218 | 0.038 | 0.016 | 0.014 | 6/35 | 98.1% |
| **0.05** | **1.00** | 35 | **94.1%** | **0.153** | 0.021 | 0.011 | 0.010 | **0/35** | **99.2%** |

*Note.* At q = 0.05 the retained round trips are shifted by a median 0.153 sd in length, 0 of 35 routes beyond the 0.25 convention: requiring full coverage drops the shortest round trips, which is the unavoidable cost of a common grid. Placing the grid at a high quantile instead shifts the analysed sample: at q = 0.95 with coverage 0.99 it keeps a median 77.0% of round trips and 93.8% of the vehicles, shifted by 0.264 sd in length, with 20 of 35 routes beyond the convention; with full coverage a median 0.1% of round trips survives.
