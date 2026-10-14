**Table S13. The four route codes that carry two services, separated vehicle by vehicle.**

The AVL records file both services of each family under one route code (the 1401 and the 1402 under 1400, which is neither). Every record was assigned to one of the two by matching it to both polylines in the direction of travel: a record is evidence for a service when it lies on a stretch that only that service drives in that direction, and each vehicle-day is split into blocks where the evidence changes service. *Shared path* is the share of each polyline that the other one also drives in the same direction; the *distinct zone* is the rest. Records that no block could assign (*unassigned*, mostly vehicle-days of a few minutes on the route) are left out of both. Month shares are over January–August 2024.

| Family (AVL code) | Main / alternate | Shared path (%) | Distinct zone (km) | Zone | AVL records | Main (%) | Alternate (%) | Unassigned (%) | Alternate share by month (%) | Vehicle-days with both |
|---|---|---|---|---|---:|---:|---:|---:|---|---|
| 1101 (1101) | 1101 / 1101-ALT (110101) | 83 / 83 | 7.9 / 8.0 | same streets, opposite direction | 2,288,740 | 52.8 | 46.8 | 0.4 | 45–49 | 39 of 3,248 |
| 1401 (1400) | 1401 / 1402 | 82 / 82 | 7.9 / 7.7 | same streets, opposite direction | 1,881,949 | 50.2 | 49.4 | 0.4 | 48–52 | 30 of 3,006 |
| 1801 (1801) | 1801 / 1801-ALT (180101) | 91 / 92 | 4.0 / 3.4 | different streets | 1,724,006 | 54.0 | 45.0 | 1.0 | 43–47 | 21 of 2,564 |
| 1802 (1802) | 1802 / 1802-ALT (180201) | 92 / 93 | 3.8 / 3.2 | different streets | 1,634,035 | 50.4 | 48.2 | 1.4 | 46–51 | 15 of 2,330 |

*Notes.* The two services of every family run side by side throughout the window (the alternate takes 43–52% of the records in every month), and a vehicle runs one service per day: at most 1.2% of the vehicle-days carry both. Where the distinct zone is the same streets driven in opposite directions, position alone cannot tell the services apart and the vehicle's heading does. The method was validated on 12 mixes of two real routes that share streets, with their route codes removed: it classified 99.0–99.9% of the records, and 99.1–99.9% of those matched the route code they carried. The variants share most of their path and their fleet, so the route-clustered bootstrap treats each family as one group (Table 9b).
