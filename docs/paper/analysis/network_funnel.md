# Network funnel — from the portal's route catalogue to the 35 processed

Derived from the three secondary sources of Section 3.3, collected from the
public web portal of UNE, Hermosillo's integrated transport system
(https://unesonora.com). Route identifiers are not routes one to one: 7 are
variants or duplicates of a base route — a variant suffix (`-ALT`, `-UI`), an
alternative branch numbered after its main route (110101 for 1101-ALT, 180101 for 1801-ALT, 180201 for 1802-ALT), or the catalogue number of a route recorded under another (2201 for 20000) — over 4 base routes (1101, 1801, 1802, 20000), so the
*base routes* column folds them in.

| Stage | Identifiers | Base routes | Source |
|---|---:|---:|---|
| Route identifiers in the portal's stop inventory | 46 | 39 | `stops_routes.json` |
| …with route geometry | 43 | 39 | `segments_route.json` |
| …with AVL records compiled for the analysis window | 35 | 32 | `records/<route>/` |
| Processed end-to-end by the pipeline | 35 | 32 | `Table 1` |

*Notes.* The three sets are strictly nested: every identifier with AVL
records also has geometry and a stop sequence, so the funnel narrows at
exactly one place — **which routes AVL records were compiled for**. The
other base routes are, in the authors' account, non-conventional services —
atypical in frequency or length, or recorded under a route code shared with
other services — that the trip-construction pipeline is not designed to
handle. Length can be checked against the geometry (below); frequency
cannot without their AVL.

- **Route identifiers in the portal's stop inventory** — ordered stop sequence per route; the widest universe the portal publishes.
- **…with route geometry** — the three without geometry are university variants: 1101-UI, 1801-UI, 1802-UI.
- **…with AVL records compiled for the analysis window** — the one cut; no identifier with AVL lacks geometry or stops. 8 of them are variants separated vehicle by vehicle from 4 route codes under which the AVL records both services.
- **Processed end-to-end by the pipeline** — every route with AVL was processed.

## The base routes without AVL, against the geometry

Polyline length of the 35 routes with AVL: median 43.5 km, range 22.4–64.8 km.

| Base route | Length (km) | Shorter than every analysed route | Variants under the same code |
|---|---:|:---:|:---:|
| 1704 | 4.8 | yes |  |
| 1003 | 5.5 | yes |  |
| 2301 | 9.0 | yes |  |
| 1803 | 13.6 | yes |  |
| 2706 | 17.7 | yes |  |
| API01 | 18.4 | yes |  |
| 322 | 40.0 |  |  |

6 of the 7 are shorter than any analysed route; the remaining 1 (322) is of ordinary length, and whether frequency or registration sets it apart is not recoverable from the files.
