
import csv
from decimal import Decimal

CSV_PATH = "data/test_simulations.csv" 
WEEK1_VAL = 1
WEEK2_VAL = 2
PMO_COL = "PMO"
W1_COL = "week_1"
W2_COL = "week_2"

def find_header_index(path, look_for="sim_id", max_scan=200):
    """Return 0-based row index of header line containing look_for; -1 if not found."""
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_scan:
                    break
                if look_for.lower() in line.lower():
                    return i
    except FileNotFoundError:
        return -1
    return -1

def run():
    header_idx = find_header_index(CSV_PATH, look_for="sim_id")
    if header_idx == -1:
        # assume first line is header
        header_idx = 0

    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as fh:
            # skip lines before header
            for _ in range(header_idx):
                next(fh)
            reader = csv.DictReader(fh)
            # basic column presence check
            required = {PMO_COL, W1_COL, W2_COL}
            missing = required - set(reader.fieldnames or [])
            if missing:
                print(f"Missing columns in CSV header: {missing}")
                return

            matched = 0
            pmo_ones = 0
            for row in reader:
                # permissive parse: ignore rows with missing week values
                try:
                    w1 = int(float(row.get(W1_COL, "") or "nan"))
                    w2 = int(float(row.get(W2_COL, "") or "nan"))
                except Exception:
                    continue
                if w1 != WEEK1_VAL or w2 != WEEK2_VAL:
                    continue
                matched += 1
                try:
                    if float(row.get(PMO_COL, "") or 0) == 1.0:
                        pmo_ones += 1
                except Exception:
                    # non-numeric PMO value -> ignore
                    pass

    except FileNotFoundError:
        print(f"File not found: {CSV_PATH}")
        return

    if matched == 0:
        print(f"No rows matched condition: {W1_COL}=={WEEK1_VAL} and {W2_COL}=={WEEK2_VAL}")
        return

    frac = Decimal(pmo_ones) / Decimal(matched)
    print(f"Rows matching condition: {matched}")
    print(f"Rows with {PMO_COL}==1 : {pmo_ones}")
    print(f"PMO fraction: {frac:.6f}")
    print(f"PMO percent : {(frac*100):.2f}%")

if __name__ == "__main__":
    run()
