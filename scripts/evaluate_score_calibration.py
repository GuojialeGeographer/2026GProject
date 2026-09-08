"""Compare raw scores, training-mean baseline and out-of-fold affine calibration.

Spatial folds are contiguous longitude bands. All fitted parameters use training rows only.
This is exploratory evaluation on an already inspected dataset, not an untouched test set.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pyrestore import load_task

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for task_file, data_name, output_name in [
        ("prs11", "shenzhen_prs11", "restorative_quality_prs11"),
        ("sixdim", "shenzhen_sixdim", "urban_perception_sixdim"),
    ]:
        task = load_task(ROOT / "tasks" / f"{task_file}.yaml")
        scores = pd.read_csv(ROOT / "outputs/submission" / output_name / "indicators.csv")
        reference = pd.read_csv(ROOT / "data" / data_name / "reference.csv")
        data = scores.merge(reference, on="capture_id", validate="one_to_one")
        fold = np.empty(len(data), dtype=int)
        for index, positions in enumerate(np.array_split(np.argsort(data.lon.to_numpy()), 5)):
            fold[positions] = index
        for predicted, observed in task["validation"]["mapping"].items():
            x, y = data[predicted].to_numpy(float), data[observed].to_numpy(float)
            calibrated, baseline = np.empty(len(y)), np.empty(len(y))
            for index in range(5):
                train, test = fold != index, fold == index
                design = np.column_stack([np.ones(train.sum()), x[train]])
                intercept, slope = np.linalg.lstsq(design, y[train], rcond=None)[0]
                calibrated[test] = np.clip(intercept + slope * x[test], 0, 1)
                baseline[test] = y[train].mean()
            for method, prediction in [("raw", x), ("training_mean", baseline), ("spatial_oof_affine", calibrated)]:
                error = prediction - y
                rows.append(dict(task=output_name, target=predicted, method=method, n=len(y),
                                 mae=np.abs(error).mean(), rmse=np.sqrt(np.mean(error**2)),
                                 r2=1 - np.sum(error**2)/np.sum((y-y.mean())**2), bias=error.mean()))
    out = ROOT / "outputs/review"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "spatial_calibration_comparison.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
