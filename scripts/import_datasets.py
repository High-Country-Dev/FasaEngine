"""Convert raw FASA web exports of ASNS and FICD into the files under ``data/``.

The exports differ from the engine's stored format:

  * ASNS: a 3-line "Target Moisture Level" preamble above title-case headers.
  * FICD: wide (one row per ingredient, display headers such as "Crude Protein (%)");
    the engine stores it long (``code,description,ingredient,quantity``) with
    snake_case parameter names, which is what ``crosswalk.json`` refers to.

PAFF is not handled here; its files are replaced as-is when it is updated.

Run from the repo root, then rebuild the pool (it reads FICD codes):
    python scripts/import_datasets.py "ASNS <date>.csv" "FICD <date>.csv"
    python scripts/build_ingredient_pool.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow `import fasa_core` without installing the package

from fasa_core.config.defaults import (  # noqa: E402  (after sys.path bootstrap)
    ASNS_FILENAME,
    DEFAULT_DATA_DIR,
    FICD_FILENAME,
)

ASNS_COLUMNS = [
    "species_code", "species", "production_system", "stage_weight", "code",
    "specification", "short_name", "unit", "restriction_type", "value",
]

# Headers whose stored name differs from what ficd_param_name() derives: renamed upstream
# since the names were fixed ("Generic", Cysteine, Chloride), or named by the original
# tool's acronym split (EAAs, PCBs). Kept so crosswalk.json and the API's composition
# keys do not change.
LEGACY_FICD_NAMES = {
    "DE Fish Carni - Generic (kcal/kg)": "de_fish_carni_old_kcal_kg",
    "DE Fish Omni - Generic (kcal/kg)": "de_fish_omni_old_kcal_kg",
    "DE Carp - Generic (kcal/kg)": "de_carp_old_kcal_kg",
    "DE Shrimp - Generic (kcal/kg)": "de_shrimp_old_kcal_kg",
    "Cysteine (%)": "cystine_percent",
    "Chloride (%)": "chlorine_percent",
    "Sum EAAs (%)": "sum_ea_as_percent",
    "Sum NEAAs (%)": "sum_nea_as_percent",
    "PCBs (ngWHOTEQ/kg)": "pc_bs_ng_whoteq_kg",
}

# The v11.1 export (release of 30 July 2026, downloaded 13 October 2026) fills
# "Iodine (mg/kg)" with the Dietary Electrolyte Balance values. While that holds, iodine
# is carried over from the FICD file being replaced; ingredients new in the export get
# no iodine value (read as 0).
IODINE_HEADER = "Iodine (mg/kg)"
ELECTROLYTE_BALANCE_HEADER = "Dietary Electrolyte Balance (mEq/Kg)"


def ficd_param_name(header: str) -> str:
    """'Dig Lys - fish (%)' -> 'dig_lys_fish_percent'."""
    if header in LEGACY_FICD_NAMES:
        return LEGACY_FICD_NAMES[header]
    s = header.replace("%", "percent")
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s)
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def convert_asns(src: Path) -> pd.DataFrame:
    df = pd.read_csv(src, encoding="utf-8-sig", skiprows=3, dtype=str)
    if len(df.columns) != len(ASNS_COLUMNS):
        raise SystemExit(f"ASNS export has {len(df.columns)} columns, expected {len(ASNS_COLUMNS)}")
    df.columns = ASNS_COLUMNS
    return df


def convert_ficd(src: Path) -> pd.DataFrame:
    wide = pd.read_csv(src, encoding="utf-8-sig", dtype=str)
    if IODINE_HEADER in wide and wide[IODINE_HEADER].equals(wide.get(ELECTROLYTE_BALANCE_HEADER)):
        previous = pd.read_csv(DEFAULT_DATA_DIR / FICD_FILENAME, dtype=str,
                               usecols=["code", "ingredient", "quantity"])
        is_iodine = previous["ingredient"] == ficd_param_name(IODINE_HEADER)
        wide[IODINE_HEADER] = wide["Code"].map(previous[is_iodine].set_index("code")["quantity"])
        print(f"WARNING: export's iodine column duplicates the electrolyte balance; kept "
              f"previous iodine ({wide[IODINE_HEADER].isna().sum()} ingredients without a "
              f"previous value left blank)")
    wide.columns = [ficd_param_name(h) for h in wide.columns]  # Code/Description -> code/description
    if wide.columns.duplicated().any():
        raise SystemExit("FICD headers collide after renaming")
    long = wide.melt(id_vars=["code", "description"], var_name="ingredient",
                     value_name="quantity", ignore_index=False)
    # one block per ingredient in export row order, parameters in export column order
    return long.sort_index(kind="stable")


def main(asns_src: str, ficd_src: str) -> int:
    asns = convert_asns(Path(asns_src))
    ficd = convert_ficd(Path(ficd_src))  # reads the FICD file it replaces, so before writing
    asns.to_csv(DEFAULT_DATA_DIR / ASNS_FILENAME, index=False)
    ficd.to_csv(DEFAULT_DATA_DIR / FICD_FILENAME, index=False)
    print(f"ASNS: {len(asns)} rows")
    print(f"FICD: {ficd['code'].nunique()} ingredients x {ficd['ingredient'].nunique()} parameters")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
