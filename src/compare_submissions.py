# Compares two submission files and reports how many labels match per task, as a reproducibility check.
import sys

import pandas as pd


# Reads two submission files and prints how many labels match per task.
def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m src.compare_submissions NEW.csv OLD.csv")
    a = pd.read_csv(sys.argv[1]).set_index("id")["label"]
    b = pd.read_csv(sys.argv[2]).set_index("id")["label"]
    assert set(a.index) == set(b.index), "the two files have different ids"
    b = b.reindex(a.index)
    for task, prefix in [("image", "img_"), ("sentiment", "text_")]:
        rows = a.index.str.startswith(prefix)
        same = int((a[rows] == b[rows]).sum())
        print(f"{task} {same / rows.sum():.2%} same")


if __name__ == "__main__":
    main()
