"""Copy the data deposit into the layout the pipeline reads (data/01_raw/).

Usage, from the repository root:

    python prepare_raw_from_deposit.py path/to/that/folder

The argument is the folder where avl_records.zip and geometry.zip of the data
deposit were extracted (the folder holding avl_records/ and
geometry/). Each avl_records/<route>.parquet becomes
data/01_raw/records/<route>/df_rec_raw_<route>_2024_01-08.parquet, and the five
files of geometry/corrected/ go to data/01_raw/. Files are copied, never moved;
existing targets are overwritten. Nothing else is read or changed.
"""
import shutil
import sys
from pathlib import Path

GEOMETRY = ("polylines.geojson", "segments.geojson", "segments_route.json",
            "stops.json", "stops_routes.json")


def main(src: Path, dst: Path) -> None:
    records = sorted((src / "avl_records").glob("*.parquet"))
    if not records:
        sys.exit(f"no avl_records/*.parquet under {src}")
    missing = [f for f in GEOMETRY if not (src / "geometry" / "corrected" / f).is_file()]
    if missing:
        sys.exit(f"missing in {src / 'geometry' / 'corrected'}: {', '.join(missing)}")

    for f in records:
        route = f.stem
        out = dst / "records" / route / f"df_rec_raw_{route}_2024_01-08.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, out)
    for name in GEOMETRY:
        shutil.copyfile(src / "geometry" / "corrected" / name, dst / name)
    print(f"{len(records)} routes and {len(GEOMETRY)} geometry files copied to {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(__file__).resolve().parent / "data" / "01_raw")
