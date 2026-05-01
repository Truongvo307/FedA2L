#!/usr/bin/env python
import os
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import polars as pl

from dataset_factory import DatasetFactory
from utils import Options, SetSeed

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)  # root directory
if ROOT not in sys.path:
    sys.path.append(ROOT)  # add ROOT to PATH
ROOT = os.path.relpath(ROOT, os.getcwd())  # relative

if __name__ == "__main__":
    total_start = time.time()
    options = Options(root=ROOT).parse_options()
    options.fix_args()
    datasetfactory = DatasetFactory(args=options.args)()
    options.update_args(
        {"num_classes": datasetfactory.num_classes, "dataset_path": datasetfactory.path}
    )
    options.display()
    options.save()
    args = options.args

    stats = defaultdict(lambda: {"min": [], "max": [], "last": []})
    time_per_experiment = []
    try:
        for t in range(args.prev, args.times):
            SetSeed(seed=args.seed + t).set()
            print(f"\n============= Running time: {t}th =============")
            print("Creating server and clients ...")
            start = time.time()
            coordinator = getattr(__import__("strategies"), args.framework)(args, t)
            coordinator.run()
            for key, value in coordinator.metrics.items():
                stats[key]["min"].append(min(value))
                stats[key]["max"].append(max(value))
                stats[key]["last"].append(value[-1])
            ts = time.time() - start
            stats["time_per_experiment"]["min"].append(ts)
            stats["time_per_experiment"]["max"].append(ts)
            stats["time_per_experiment"]["last"].append(ts)
    except KeyboardInterrupt:
        shutil.rmtree(args.save_path)

    rows = []
    for metric, stats in stats.items():
        row = {
            "metric": metric,
            "avg_min": np.mean(stats["min"]),
            "std_min": np.std(stats["min"]),
            "avg_max": np.mean(stats["max"]),
            "std_max": np.std(stats["max"]),
            "avg_last": np.mean(stats["last"]),
            "std_last": np.std(stats["last"]),
        }
        rows.append(row)
    stats = pl.DataFrame(rows)
    stats.write_csv(os.path.join(args.save_path, "results.csv"))
    print(stats)
