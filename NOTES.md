# Engineering notes

Decisions and gotchas worth keeping, grouped by area. Headline design
write-ups live in [`DESIGN.md`](./DESIGN.md); this file is the
"things-I-learned-the-hard-way" log.

## Data & features

- **Entity = `nameDest`, not `nameOrig`.** EDA found 99.85 % of originators
  appear exactly once, so per-customer window aggregates are near-empty.
  Destinations (mule accounts) repeat — mean 2.34, max 113 — so their
  aggregates carry the real signal.
- **Time-based split on `step`.** Random splits leak future transactions
  into training. Fraud density also rises ~4× over time — *only visible*
  because of this split.
- **Feast point-in-time joins.** For each training row at time T, Feast
  returns only features computed before T. The leakage prevention you'd
  do by hand becomes a store-enforced guarantee, and the same definitions
  feed serving — eliminating train/serve skew.
- **Great Expectations is on 1.x.** The brief assumed 0.18; the API
  changed substantially. Validation code is pinned and tested.

## Model

- **AUC-PR over AUC-ROC.** At 0.13 % fraud rate, ROC is dominated by
  trivial true-negatives and looks deceptively perfect.
- **`scale_pos_weight` + threshold tuning, not SMOTE.** SMOTE's ranking
  gain over weighting was ~0.002 AUC-PR on already-separable data —
  not worth the synthetic-data tax. Threshold tuning is the orthogonal
  lever for production alert workload.
- **Optuna TPE + MLflow nested runs.** Each trial is a nested run under
  the parent HPO run, so the registry view stays clean.

## Serving

- **Tree-path-dependent SHAP for serving.** The interventional explainer
  (100-row background) costs ~100× more per request. Switching dropped
  P99 from 1,800 ms → 21 ms with no decision change, only a minor and
  still-valid shift in attribution method.
- **MLflow pinned to 2.x.** MLflow 3 removed stage-based Model Registry
  (`Staging → Production`); the serving path loads by stage.
- **Multi-stage Dockerfile, non-root.** Build deps stay in the builder
  stage; runtime image is slim.

## Infra & gotchas

- **macOS port 5000 is AirPlay.** MLflow remapped to host 5050
  (`5050:5000` in docker-compose).
- **`localhost` inside a container means *the container*.** Feast
  `feature_store.yaml` uses `${REDIS_CONNECTION_STRING}` —
  `localhost:6379` on host, `host.docker.internal:6379` in containers,
  `host.minikube.internal:6379` in k8s. Feast does **not** support
  `:-default` shell-style fallback.
- **`.env` without trailing newline + `>>` appends merge lines.**
  Caused a `KeyError: 'port'` in Feast because `${REDIS_CONNECTION_STRING}`
  resolved to an empty string. Now appended with explicit newlines.
- **xgboost on macOS needs libomp.** Conda-forge package name is
  `llvm-openmp`, not `libomp`.
- **MLflow retry storm when the server is down.** Validate + retrain
  trigger do a 2-second TCP reachability probe first, then fail fast.

## CI / CD

- **GHCR rejects mixed-case owners.** `cd.yml` lowercases
  `github.repository_owner` via `tr '[:upper:]' '[:lower:]'` before
  composing the image tag.
- **Pre-commit isolated envs drift from local.** All three quality hooks
  (ruff, black, mypy) use `language: system` so a local fix and a CI
  fix are the same fix.
- **GitHub Actions uses `setup-miniconda`.** Mirrors `environment.yml`
  exactly, so version pins in one place govern both local and CI.

## Language / framework footguns

- **`# type: one of …` is a PEP 484 type comment.** mypy will try to
  parse it. Renamed to `# The "type" column: …` to avoid the collision.
- **`from __future__ import annotations` breaks Kubeflow Pipelines.**
  PEP 563 makes annotations strings; KFP introspects at decorate time
  and reports `Got: str`. Removed the future-import from component files.
- **Evidently 0.7 rewrote its API end-to-end.** Brief assumed 0.4 —
  reports, presets, and metric names all moved.
- **`/metrics` 307-redirects to `/metrics/`** when prometheus-client is
  mounted as a sub-app in FastAPI. Prometheus scrape path uses the
  trailing slash.
