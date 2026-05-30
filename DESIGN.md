# Design Decisions

## Dataset

PaySim synthetic mobile-money transactions: 6,362,620 rows, 11 columns, one simulated
month (`step` = hours since start, 0–743). Label: `isFraud`.

## Exploratory Data Analysis (Week 2)

**Q1 — Fraud rate.** 8,213 / 6,362,620 = **0.1291%** (~1 in 775). Extreme imbalance.
Consequences: accuracy is meaningless; use **AUC-PR** as the primary metric (not AUC-ROC);
apply `scale_pos_weight`; tune the decision threshold instead of defaulting to 0.5.

**Q2 — Fraud by type.** Fraud occurs _only_ in `TRANSFER` (rate 0.77%) and `CASH_OUT`
(0.18%); `PAYMENT`/`CASH_IN`/`DEBIT` have zero fraud. Mechanism: TRANSFER to a mule, then
CASH_OUT. Production scores only these two types; type is known at request time (no leakage).

**Q3 — Time of day.** Fraud count is ~constant hourly while legit volume is diurnal, so the
fraud _rate_ spikes to ~22% at 04:00–05:00 vs ~0.07% midday. `hour_of_day` (step % 24) is
predictive. Caveat: the flat hourly fraud count is a simulation artifact.

**Q4 — Amount.** Fraud is far larger: median 441,423 vs 74,685 (legit); capped at exactly
10,000,000 (PaySim transfer limit). `amount` and `amount_to_balance_ratio` are predictive.

**Q5 — Nulls / duplicates.** None — data is clean. These invariants become Great
Expectations rules (Week 3) to guard against dirty production data.

**Q6 — Transaction-time vs. history-dependent features.**

- _Available at request time (no leakage):_ amount, type, balances, hour_of_day,
  balance_delta, amount_to_balance_ratio, is_round_amount, errorBalanceOrig/Dest.
- _Require history:_ tx_count_1h/24h/7d, tx_amount_sum/avg_24h.
- Aggregating these over the full dataset leaks future info → invalid model. Fixed via
  point-in-time-correct joins (Feast).

## Feature engineering — revision to the original brief

**Finding:** 99.85% of originating customers (`nameOrig`) appear exactly once
(mean 1.0015 tx). Per-customer window aggregates are therefore near-empty.

**Decisions:**

1. Primary predictive features are **row-level balance logic**, not customer windows.
   Strongest signal: `errorBalanceDest` (fraud avg 745,139 vs 92,757 legit, ~8×);
   `errorBalanceOrig` ≈ 0 for fraud (origin cleanly emptied).
2. Window aggregates are keyed on **`nameDest`** (mules repeat: mean 2.34, max 113),
   not `nameOrig`.
3. Window features are retained mainly to demonstrate the PySpark + Feast
   point-in-time-join engineering, with their limited predictive value documented honestly.

## Metric & validation decisions

- Primary metric: AUC-PR. Secondary: precision at 90% recall.
- Time-based train/test split (first 80% of `step` = train) — never random, since fraud
  patterns evolve over time and a random split leaks temporal information.

## Class-imbalance strategy (Week 9)

Compared three approaches on the held-out test set (one MLflow experiment, `imbalance-comparison`):

| Strategy          | AUC-PR              | precision @ 90% recall                   |
| ----------------- | ------------------- | ---------------------------------------- |
| scale_pos_weight  | 0.998               | 0.996                                    |
| SMOTE (ratio 0.1) | 0.9999              | 1.000                                    |
| threshold tuning  | (ranking unchanged) | precision 0.996 @ recall 0.91, thr=0.999 |

**Decision: keep `scale_pos_weight` + threshold tuning; skip SMOTE.**

- SMOTE's ranking gain over `scale_pos_weight` is negligible (~0.002 AUC-PR) on PaySim's
  already-separable data — not worth the added pipeline complexity and synthetic-data risk.
- Threshold tuning is orthogonal to the model: it doesn't change AUC-PR, it selects the
  operating point. Because the model is extremely confident, hitting the 90%-recall business
  target requires a ~0.999 threshold, yielding ~99.6% precision. This is the lever that
  actually governs the fraud-ops alert workload in production.

## Serving performance (Week 18 — Locust, 50 concurrent users, 60s)

| Metric | Value |
| --- | --- |
| Requests | 8,945 (0 failures) |
| Throughput | ~149 req/s |
| Latency P50 | 8 ms |
| Latency P95 | 14 ms |
| Latency P99 | **21 ms** |

**Key tuning:** the per-prediction SHAP explainer was switched from *interventional*
(100-row background) to *tree-path-dependent* (no background). This cut per-request SHAP
cost ~50-100x and dropped P99 from ~1,800 ms to 21 ms (and throughput 41 -> 149 req/s) with
no change to the decision — only a minor, still-valid shift in attribution method. Well
under the 100 ms P99 target.

## End-to-end integration (Week 22)

`scripts/integration_smoke.sh` (also `make integration`) brings up the full stack and
exercises it for a short cycle:

1. `docker compose up -d` (Redpanda, MinIO, Redis, MLflow, Prometheus, Grafana)
2. `uvicorn src.serving.main:app` (serving API on :8000)
3. Streaming consumer (real-time Redis counters)
4. Producer: 1,000 PaySim transactions at 100 TPS into `transactions.raw`
5. Locust: 10 users, 30s, light load on `/predict`
6. Verify Prometheus has scraped `fraud_predictions_total > 0`
7. Verify Redis has `dest:*` live counters from the consumer

Exits 0 on success; logs land in `/tmp/fraud-integration/`. Used as a pre-record sanity
check before a portfolio screencast.
