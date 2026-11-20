# Builds the compute-cost table: fit time, predict time and model size of the best run per model.
import json

import pandas as pd

from src import config


DATASETS = ("image", "sentiment")
MODELS = ("knn", "tree", "forest", "nn")
REQUIRED_FIELDS = (
    "dataset",
    "model",
    "run_id",
    "transform",
    "train_acc",
    "val_acc",
    "fit_seconds",
    "predict_seconds",
    "model_bytes",
)


# Finds the best validation run of one model and returns its cost fields.
def best_logged_run(dataset, model):
    model_dir = config.RUNS_DIR / dataset / model
    records = []
    for config_path in model_dir.glob("*/config.json"):
        record = json.loads(config_path.read_text(encoding="utf-8"))
        if "val_acc" not in record:
            continue
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(
                f"{config_path} is missing required fields: {', '.join(missing)}"
            )
        if record["dataset"] != dataset or record["model"] != model:
            raise ValueError(f"{config_path} has mismatched dataset/model metadata")
        records.append((record, config_path))

    if not records:
        raise FileNotFoundError(
            f"No logged validation runs found for {dataset}/{model}"
        )

    record, config_path = min(
        records,
        key=lambda item: (
            -float(item[0]["val_acc"]),
            float(item[0]["fit_seconds"]),
            item[0]["run_id"],
        ),
    )
    return {
        "dataset": dataset,
        "model": model,
        "run_id": record["run_id"],
        "transform": record["transform"],
        "train_acc": record["train_acc"],
        "val_acc": record["val_acc"],
        "fit_seconds": record["fit_seconds"],
        "predict_seconds": record["predict_seconds"],
        "model_bytes": record["model_bytes"],
        "model_size_mib": record["model_bytes"] / (1024 ** 2),
        "config_path": str(config_path.relative_to(config.ROOT)),
    }


# Collects the best run per model and dataset, saves the table and prints it.
def main():
    rows = [
        best_logged_run(dataset, model)
        for dataset in DATASETS
        for model in MODELS
    ]
    table = pd.DataFrame(rows)
    output = config.ROOT / "results" / "compute_cost.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False, formatters={
        "train_acc": "{:.4f}".format,
        "val_acc": "{:.4f}".format,
        "fit_seconds": "{:.3f}".format,
        "predict_seconds": "{:.3f}".format,
        "model_size_mib": "{:.2f}".format,
    }))


if __name__ == "__main__":
    main()
