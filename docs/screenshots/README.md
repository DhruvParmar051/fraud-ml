# Screenshots

Drop the following PNGs here and they'll render in the top-level README.

| Filename | What to capture |
| --- | --- |
| `locust_p99.png` | Locust web UI or `reports/locust_report.html` showing **P99 < 100 ms** at 50 users / 60s |
| `grafana_dashboard.png` | Grafana *Fraud Detection* dashboard with traffic flowing — request rate + latency panels |
| `mlflow_runs.png` | MLflow UI run list for the `fraud-detection-hpo` experiment (≥30 runs, sorted by `val_auc_pr`) |
| `mlflow_registry.png` | MLflow Model Registry — `fraud-detector` with a version in **Production** stage |
| `wandb_sweep.png` | W&B sweep parallel-coordinates plot showing the HPO search |
| `evidently_drift.png` | An Evidently drift report excerpt — overall drift score + the drifted-features table |

Capture at 1600px width minimum so the README renders crisp on Retina.
