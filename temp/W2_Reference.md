# Data Work (ETL + EDA)

## Outcomes

By the end of this reference, you can:

* Build an **offline-first** data workflow: **load → verify → clean → transform → analyze → visualize → conclude**.
* Read/write common formats (CSV/JSON/Parquet), manage **schemas/types**, and keep **raw data immutable**.
* Apply practical data cleaning: **missingness**, **duplicates**, **type fixes**, **text normalization**, **datetime parsing**, **outliers**.
* Use pandas effectively for real work: **selection**, **groupby/agg**, **merge (validated)**, **reshape (melt/pivot)**.
* Prevent common join disasters: detect **join explosions**, enforce **key uniqueness**, and validate joins.
* Perform practical EDA with **descriptive stats**, **rates/ratios**, and **comparison thinking** (incl. bootstrap intervals).
* Create clear charts using **one plotting library**, with consistent chart anatomy (figure, axes, marks, scales, labels).
* Produce job-ready handoffs: **processed datasets**, **EDA notebook**, **figures**, and a **written summary with caveats**.
* Ship a small ETL pipeline with **pure transforms**, **config**, **logging**, **QA checks**, and **idempotent outputs**.

---

## Tool stack

Opinionated minimal stack (high ROI, low sprawl):

* **pandas**
  For tabular loading/cleaning/joins/reshaping/EDA. Avoid re-implementing table operations manually; use pandas’ battle-tested methods.
* **pyarrow** (via pandas Parquet)
  For **Parquet** read/write (fast, typed, compact). Avoid “CSV-only” workflows once data is stabilized.
* **plotly** (**plotly.express** + `fig.update_*`)
  For interactive, publishable charts and easy exporting. Avoid mixing multiple plotting libraries.
* **httpx** (already known)
  For API extraction. Avoid “live calls” as a dependency; always cache responses offline.
* **typer** (optional)
  For a small CLI wrapper around ETL runs. Avoid complex CLI frameworks.
* **logging** (stdlib)
  For run visibility and auditability. Avoid print-only pipelines.
* **duckdb**
  For lightweight analytics SQL on local files (CSV/Parquet) when it’s simpler than pandas. Avoid heavyweight DB setup.
* **PostgreSQL** (not introduced here)
  Industry-standard production relational DB for SQL fundamentals + multi-user persistence (schemas/constraints/indexes).

> **Not in the core stack**: seaborn/matplotlib teaching, pydantic-heavy validation, big workflow orchestrators, distributed compute. Keep it small and shippable.

---

## Project conventions

### Folder layout (opinionated, offline-first)

```
project/
  pyproject.toml
  README.md

  data/
    raw/            # immutable inputs (never edited)
    cache/          # API responses / intermediate downloads (safe to delete)
    processed/      # clean, analysis-ready outputs (idempotent)
    external/       # provided reference data (manual drops)

  notebooks/        # EDA + exploration (read processed/)
  reports/
    figures/        # exported images for writeups/slides
    summary.md      # short written findings + caveats

  src/
    project_name/
      __init__.py
      config.py
      etl.py
      io.py
      quality.py
      transforms.py
      viz.py
      utils.py

  scripts/
    run_etl.py       # thin entrypoint (optional if using `python -m`)
```

### Conventions that prevent pain

* **Raw data is immutable**: never modify files under `data/raw/`.
* **Idempotent processed outputs**: rerunning ETL yields the same outputs given same inputs/config.
* **One source of truth for paths**: use `pathlib.Path` and centralize in config.
* **Determinism**: pin seeds for simulations (bootstrap), and record config + git commit when possible.
* **Schema-aware**: enforce dtypes after load; never “let pandas guess” silently for critical fields.
* **Fail fast**: validate assumptions early (unique keys, ranges, required columns).
* **Separation of concerns**: I/O (load/save) ≠ transforms ≠ analysis/viz.

---

## Workflow-first structure

**Canonical flow (repeat every project):**

1. **Load** (from raw/cache)
2. **Verify** (columns, types, key uniqueness, row counts, missingness)
3. **Clean** (types, missing, duplicates, normalization)
4. **Transform** (joins, reshape, feature engineering)
5. **Analyze** (stats tables, comparisons, bootstraps)
6. **Visualize** (charts with titles/labels + export)
7. **Conclude** (written summary + caveats + next questions)

Keep notebooks focused on steps **4–7** using `data/processed/` as inputs.

---

# 1) Data sources & extraction

**What it’s for:** Get data into your project reliably, even with flaky internet.

### Canonical operations / patterns

* Prefer **local-first** sources: CSV/JSON drops, exported reports, sample API responses.
* If using APIs: **cache responses** and optionally store a small **sample**.
* Track extraction metadata: **timestamp, endpoint, parameters, status code** (minimally).

### Pitfalls + checks

* **Changing upstream data** → your results change. Cache and pin snapshots.
* **API pagination** → partial data if ignored.
* **Rate limits / failures** → silent partial files if you don’t validate sizes.
* **Encoding** issues for CSV/text; always declare or detect.

### Minimal code templates

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import time
import httpx

@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    cache: Path
    processed: Path

def make_paths(root: Path) -> Paths:
    return Paths(
        root=root,
        raw=root / "data" / "raw",
        cache=root / "data" / "cache",
        processed=root / "data" / "processed",
    )

def fetch_json_cached(url: str, cache_path: Path, *, ttl_s: int | None = None) -> dict:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and ttl_s is not None:
        age = time.time() - cache_path.stat().st_mtime
        if age < ttl_s:
            return json.loads(cache_path.read_text(encoding="utf-8"))

    if cache_path.exists() and ttl_s is None:
        # offline-first default: reuse cached if present
        return json.loads(cache_path.read_text(encoding="utf-8"))

    with httpx.Client(timeout=20.0) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()

    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data
```

---

# 2) Loading data (I/O) with pandas

**What it’s for:** Bring data into DataFrames with predictable schema and minimal surprises.

### Canonical operations / patterns

* Centralize I/O in `src/.../io.py`.
* Read CSV with explicit options:

  * `dtype=` for known columns (strings vs numeric)
  * `na_values=` for custom missing markers
  * `encoding=`, `sep=`, `decimal=` as needed
* Prefer **Parquet** for processed outputs:

  * preserves dtypes
  * faster read/write
  * smaller files

### Pitfalls + checks

* Pandas **dtype inference lies** (IDs become numbers, leading zeros lost).
* `parse_dates=True` without a plan → timezone and invalid parsing issues.
* CSV quoting/escape issues → weird column shifts. Validate column count.

### Minimal code templates

```python
from pathlib import Path
import pandas as pd

def read_csv_typed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "user_id": "string",
            "order_id": "string",
        },
        na_values=["", "NA", "N/A", "null", "None"],
        keep_default_na=True,
    )
    return df

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
```

---

# 3) Schema, dtypes, and “manual → pandas” bridge

**What it’s for:** Treat data types as a core part of correctness; reduce “pandas magic” with a tiny mental bridge.

### Manual → pandas bridge (tiny)

```python
# manual (list of dicts) → DataFrame
rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
df = pd.DataFrame(rows)

# DataFrame → manual (records)
records = df.to_dict(orient="records")
```

### Canonical operations / patterns

* **IDs are strings** unless you truly compute on them.
* Use pandas’ nullable types where appropriate:

  * `"string"`, `"Int64"`, `"boolean"`, `"Float64"`
* Use explicit casting in a dedicated transform step.

### Pitfalls + checks

* Numeric IDs: `00123` becomes `123` → unrecoverable.
* Mixed types in a column → slow ops and subtle bugs.
* Float “integers” with missing values → should be `"Int64"`.

### Minimal code templates

```python
import pandas as pd

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        user_id=df["user_id"].astype("string"),
        amount=pd.to_numeric(df["amount"], errors="coerce").astype("Float64"),
        quantity=pd.to_numeric(df["quantity"], errors="coerce").astype("Int64"),
    )
```

---

# 4) Data quality checks (fail fast)

**What it’s for:** Turn assumptions into checks so bad data fails loudly early.

### Canonical operations / patterns

* Column presence + non-empty dataset.
* Key integrity: uniqueness and null checks.
* Range checks (e.g., non-negative amounts).
* Basic distribution sanity: suspicious row counts, extreme missingness.

### Pitfalls + checks

* “Looks fine” isn’t a check; turn it into a function.
* Don’t overbuild validation frameworks; keep it lightweight.
* Validate **before** joins and **after** major transforms.

### Minimal code templates

```python
from __future__ import annotations
import pandas as pd

def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

def assert_non_empty(df: pd.DataFrame, name: str = "df") -> None:
    assert len(df) > 0, f"{name} has 0 rows"

def assert_unique_key(df: pd.DataFrame, key: str, *, allow_na: bool = False) -> None:
    if not allow_na:
        assert df[key].notna().all(), f"{key} contains NA"
    dup = df[key].duplicated(keep=False) & df[key].notna()
    assert not dup.any(), f"{key} is not unique; {dup.sum()} duplicate rows"

def assert_in_range(s: pd.Series, lo=None, hi=None, name: str = "value") -> None:
    if lo is not None:
        assert (s >= lo).all(), f"{name} below {lo}"
    if hi is not None:
        assert (s <= hi).all(), f"{name} above {hi}"
```

---

# 5) Missingness, duplicates, and basic cleaning

**What it’s for:** Make data usable: handle gaps, de-dup, normalize text, and remove obvious noise without inventing facts.

## Missingness

### Canonical operations / patterns

* Measure missingness **per column** and **per row**.
* Decide: **drop**, **impute**, or **flag**.

  * For analytics EDA, prefer: **flag + cautious impute** only when justified.
* Use domain meaning: “missing” might mean “not applicable” vs “unknown”.

### Pitfalls + checks

* Blanket `.dropna()` destroys data and biases results.
* Filling missing with 0 can create fake events.
* Missingness might correlate with cohorts (systematic bias).

### Templates

```python
import pandas as pd

def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    return (
        df.isna().sum()
          .rename("n_missing")
          .to_frame()
          .assign(p_missing=lambda t: t["n_missing"] / n)
          .sort_values("p_missing", ascending=False)
    )

def add_missing_flags(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[f"{c}__isna"] = out[c].isna()
    return out
```

## Duplicates

### Canonical operations / patterns

* Define duplicates by **business keys** (not whole-row equality).
* Keep the “best” record: latest timestamp, most complete, highest priority source.

### Pitfalls + checks

* Dropping duplicates without a rule can delete real distinct events.
* Duplicates often come from joins or repeated extracts.

### Templates

```python
def dedupe_keep_latest(df: pd.DataFrame, key_cols: list[str], ts_col: str) -> pd.DataFrame:
    return (
        df.sort_values(ts_col)
          .drop_duplicates(subset=key_cols, keep="last")
          .reset_index(drop=True)
    )
```

---

# 6) Text normalization and categorical hygiene

**What it’s for:** Make categories analyzable; reduce “same thing spelled 5 ways” issues.

### Canonical operations / patterns

* Normalize: trim, casefold, whitespace collapse.
* Map common synonyms and typos with a controlled dictionary.
* Keep original raw text columns when feasible (`*_raw`).

### Pitfalls + checks

* Over-normalizing loses meaning (e.g., product codes).
* Aggressive regex can merge distinct categories.

### Minimal code templates

```python
import re
import pandas as pd

_ws = re.compile(r"\s+")

def normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .str.strip()
         .str.casefold()
         .str.replace(_ws, " ", regex=True)
    )

def apply_mapping(s: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return s.map(lambda x: mapping.get(x, x))
```

---

# 7) Datetime parsing and time zones

**What it’s for:** Make time-based analysis correct: trends, cohorts, durations, and ordering.

### Canonical operations / patterns

* Use `pd.to_datetime(..., errors="coerce")` with explicit assumptions.
* Store:

  * event timestamp as `datetime64[ns]` (ideally timezone-aware if needed)
  * date as `.dt.date` only for grouping/reporting
* Extract time parts: day/week/month, hour, dayofweek (careful with locale).

### Pitfalls + checks

* Ambiguous formats (MM/DD vs DD/MM) → wrong dates silently.
* Timezones: mixing naive and aware timestamps.
* Sorting strings that look like dates.

### Minimal code templates

```python
import pandas as pd

def parse_datetime(df: pd.DataFrame, col: str, *, utc: bool = True) -> pd.DataFrame:
    dt = pd.to_datetime(df[col], errors="coerce", utc=utc)
    return df.assign(**{col: dt})

def add_time_parts(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    ts = df[ts_col]
    return df.assign(
        date=ts.dt.date,
        year=ts.dt.year,
        month=ts.dt.to_period("M").astype("string"),
        dow=ts.dt.day_name(),
        hour=ts.dt.hour,
    )
```

---

# 8) Outliers and sanity checks (practical)

**What it’s for:** Identify and manage extreme values that distort summaries and charts.

### Canonical operations / patterns

* Start with **IQR** and **percentiles** (e.g., p1/p99).
* Prefer **robust stats**: median, IQR, trimmed means.
* Decide: keep (but cap for visualization), filter with justification, or flag.

### Pitfalls + checks

* Removing “outliers” can delete the most valuable events (fraud, VIPs, spikes).
* Outliers can be data errors (unit mismatch) → check data lineage.

### Minimal code templates

```python
import pandas as pd

def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)

def winsorize(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(lower=a, upper=b)
```

---

# 9) Core pandas operations that matter in practice

**What it’s for:** Get fluent with the small set of operations that cover most data work.

## Selection & assignment

### Canonical patterns

* Use `.loc[row_filter, cols]` for clarity.
* Use `.assign(...)` for chainable transforms.
* Use vectorized `.str`, `.dt`, `.where`, `.clip`, `.fillna`.

### Pitfalls + checks

* Chained indexing (`df[df.a>0]["b"]=...`) → bugs. Use `.loc` / `.assign`.
* Overuse `apply` on rows; it’s slow and often unnecessary.

### Templates

```python
df2 = (
    df.loc[df["status"].eq("paid"), ["user_id", "amount", "created_at"]]
      .assign(amount_usd=lambda d: d["amount"] * 1.0)  # placeholder
)
```

## Groupby/aggregate

### Canonical patterns

* Aggregate with named outputs.
* Compute rates/ratios with explicit numerators/denominators.

### Pitfalls + checks

* Aggregating on dirty keys (case/whitespace) duplicates groups.
* Division by zero; define behavior.

### Templates

```python
summary = (
    df.groupby("country", dropna=False)
      .agg(
          n=("order_id", "size"),
          revenue=("amount", "sum"),
          aov=("amount", "mean"),
          med_amount=("amount", "median"),
      )
      .reset_index()
)

# rates with explicit components
rate = (
    df.assign(is_refund=df["status"].eq("refund"))
      .groupby("country")
      .agg(refunds=("is_refund", "sum"), total=("is_refund", "size"))
      .assign(refund_rate=lambda t: t["refunds"] / t["total"])
      .reset_index()
)
```

## Reshape (tidy data)

### Tidy data mental model

* Each variable is a column; each observation is a row; each type of observational unit is a table.
* **Long** form: one value column + category columns → best for plotting/groupby.
* **Wide** form: categories become columns → best for human readability and some exports.

### Canonical operations

* `melt` to go **wide → long**
* `pivot_table` to go **long → wide**

### Pitfalls + checks

* Pivoting without unique keys creates accidental aggregation.
* Wide tables with many columns are fragile and harder to validate.

### Templates

```python
long = df.melt(
    id_vars=["user_id", "date"],
    value_vars=["clicks", "views"],
    var_name="metric",
    value_name="value",
)

wide = long.pivot_table(
    index=["user_id", "date"],
    columns="metric",
    values="value",
    aggfunc="sum",
).reset_index()
```

---

# 10) Joining data correctly (and safely)

**What it’s for:** Combine tables without silently duplicating, dropping, or corrupting records.

### Canonical operations / patterns

* Always:

  * name keys clearly (`user_id`, `order_id`)
  * validate join cardinality with `validate=...`
  * use explicit `suffixes=...`
* Pre-check uniqueness on the “one” side in 1:m joins.
* After join, verify expected row counts and null patterns in joined columns.

### Common join types (operational)

* `how="left"`: keep main table, enrich with lookup data (most common for analytics)
* `how="inner"`: only matched rows (can drop data unexpectedly)
* `how="outer"`: reconciliation and completeness checks

### Pitfalls + checks (put these *before* merge)

* **Join explosion**: keys not unique on either side → row count multiplies.
* **Key dtype mismatch**: `"001"` vs `1` → massive missing matches.
* **Many-to-many when you assumed one-to-many** → corrupted aggregates.

### Minimal code templates

```python
import pandas as pd

def safe_left_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    *,
    validate: str,
    suffixes: tuple[str, str] = ("", "_r"),
) -> pd.DataFrame:
    return left.merge(right, how="left", on=on, validate=validate, suffixes=suffixes)

# Example: orders (many) -> users (one)
# validate="many_to_one" means: left can repeat keys; right must be unique.
orders_enriched = safe_left_join(
    orders,
    users,
    on="user_id",
    validate="many_to_one",
    suffixes=("", "_user"),
)

# Quick join sanity
assert len(orders_enriched) == len(orders), "Row count changed on left join (join explosion?)"
```

**Operational join validation checklist**

* [ ] Are join keys present and **non-null** where required?
* [ ] Do both sides have the **same dtype** for keys?
* [ ] Is the “one” side actually **unique**?
* [ ] Did row count change unexpectedly?
* [ ] Are new columns missing too often (match rate suspiciously low)?

---

# 11) Transformation design: pure functions + piping

**What it’s for:** Make transformations readable, testable, and reusable across notebooks and ETL scripts.

### Canonical operations / patterns

* Write transforms as **pure functions**: `df -> df`.
* Use `.pipe()` to build clean pipelines.
* Keep I/O outside transforms.
* Split steps: `clean_*`, `add_*`, `join_*`, `reshape_*`.

### Pitfalls + checks

* “One giant notebook cell” transforms are untestable.
* Hidden global state (paths/config) inside transforms.

### Minimal code templates

```python
import pandas as pd

def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.rename(columns=str.lower)
          .pipe(enforce_schema)
          .pipe(parse_datetime, col="created_at", utc=True)
          .assign(status=lambda d: normalize_text(d["status"]))
    )

def build_analytics_table(orders: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    return (
        orders.pipe(clean_orders)
              .pipe(lambda d: safe_left_join(d, users, on="user_id", validate="many_to_one"))
              .pipe(add_time_parts, ts_col="created_at")
    )
```

---

# 12) Feature engineering for analytics (not ML)

**What it’s for:** Create interpretable derived fields for grouping, slicing, and reporting.

### Canonical operations / patterns

* **Ratios**: conversion rate, refund rate, revenue per user.
* **Bins**: price buckets, tenure buckets.
* **Time parts**: week, month, hour, day-of-week.
* **Cohorts (optional)**: first-seen month, retention curves.

### Pitfalls + checks

* Ratios can mislead: always show numerator/denominator.
* Bins hide detail; choose bin edges intentionally.
* Cohorts require consistent “first event” definition.

### Minimal code templates

```python
import pandas as pd

def add_ratio(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        revenue_per_item=lambda d: d["amount"] / d["quantity"].where(d["quantity"] > 0)
    )

def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        amount_bin=pd.cut(
            df["amount"],
            bins=[0, 10, 50, 100, 500, float("inf")],
            right=False,
            include_lowest=True,
        )
    )

def add_first_seen_cohort(df: pd.DataFrame) -> pd.DataFrame:
    # expects columns: user_id, created_at
    first = df.groupby("user_id")["created_at"].min().rename("first_seen")
    out = df.join(first, on="user_id")
    return out.assign(cohort=lambda d: d["first_seen"].dt.to_period("M").astype("string"))
```

---

# 13) Practical EDA workflow (tables → comparisons → caveats)

**What it’s for:** Turn a dataset into decisions: “What happened?”, “Where?”, “Compared to what?”, “How sure are we?”

### Canonical operations / patterns

* Start with **data audit**:

  * row count, columns, dtypes
  * missingness report
  * key uniqueness
* Define a small set of **business questions** (3–6) and answer them with:

  * summary tables
  * 1–2 charts each
  * a short interpretation + caveat
* Use **comparison thinking**:

  * compare cohorts, segments, time windows
  * report **absolute** and **relative** differences (effect sizes)

### Pitfalls + checks

* Overfitting conclusions to noisy small groups → show sample sizes.
* Confusing correlation with causation → explicitly state limitations.
* “Average” hides skew → include median and percentiles.

### Minimal templates

```python
import pandas as pd

def describe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    return pd.Series({
        "n": s.notna().sum(),
        "mean": s.mean(),
        "median": s.median(),
        "p25": s.quantile(0.25),
        "p75": s.quantile(0.75),
        "p90": s.quantile(0.90),
        "min": s.min(),
        "max": s.max(),
    })

def compare_groups(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    g = df.groupby(group_col)[value_col]
    return g.agg(n="size", mean="mean", median="median").reset_index()
```

---

# 14) Practical uncertainty: bootstrap intervals (simulation-based)

**What it’s for:** Quantify uncertainty without heavy theory; answer “How variable is this estimate?”

### When to use

* You’re comparing means/medians/rates between groups.
* You want a rough confidence interval without relying on assumptions.

### Pitfalls + checks

* Bootstrap assumes your sample represents the population reasonably.
* Very small n → bootstrap intervals can be unstable.
* Keep random seed fixed for reproducibility in reports.

### Minimal code template (mean/rate difference)

```python
from __future__ import annotations
import numpy as np
import pandas as pd

def bootstrap_diff_means(a: pd.Series, b: pd.Series, *, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    assert len(a) > 0 and len(b) > 0, "Empty group after cleaning"

    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(sa.mean() - sb.mean())
    diffs = np.array(diffs)

    return {
        "diff_mean": float(a.mean() - b.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
    }
```

**Effect size intuition (operational)**

* Always include:

  * **absolute difference** (e.g., +$2.10 AOV)
  * **relative difference** (e.g., +8%)
  * sample sizes
* Beware “statistically notable” but practically irrelevant differences.

> Optional (operational-only): If you use a statistical test, state: the question it answers, assumptions, and failure modes (non-independence, non-stationarity, multiple comparisons).

---

# 15) Visualization with Plotly (one library)

**What it’s for:** Communicate patterns and comparisons clearly, with consistent chart anatomy and minimal fuss.

## Universal chart components (library-agnostic)

* **Figure**: the container for the whole chart (layout + data)
* **Axes**: coordinate system (x/y), scales, ticks
* **Marks/Traces**: points/lines/bars/areas (the actual data encoding)
* **Scales**: mapping data → visual (linear/log, categorical ordering)
* **Legend**: mapping color/symbol to categories
* **Labels/Titles**: axis labels, chart title, subtitles/captions
* **Annotations**: callouts (thresholds, events)
* **Facets/Subplots**: small multiples to compare groups consistently

## Chart choice guidance (decision patterns)

Use the simplest chart that answers the question:

* **Compare categories** → bar (sorted), dot plot
* **Trend over time** → line (with smoothing only if stated)
* **Distribution** → histogram, box/violin (use sparingly), ECDF (optional later)
* **Relationship (2 numeric)** → scatter (+ trendline optional)
* **Composition** → stacked bars (avoid pie for dense comparisons)
* **Rates vs volume** → dual views (two charts) instead of dual axes (usually)

**Operational rules**

* Sort categories by value for readability.
* Always show **units** and **time window** in titles or captions.
* Include **n** where comparisons could be misleading.

## Canonical Plotly patterns

* Use `plotly.express` for quick, consistent charts.
* Customize via `fig.update_layout(...)` and `fig.update_xaxes/yaxes(...)`.
* Facet with `facet_col` / `facet_row`.
* Export images to `reports/figures/` (requires Kaleido installed).

### Minimal code templates

```python
from pathlib import Path
import plotly.express as px

def save_fig(fig, path: Path, *, scale: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), scale=scale)  # needs `kaleido`

def bar_sorted(df, x, y, title: str):
    d = df.sort_values(y, ascending=False)
    fig = px.bar(d, x=x, y=y, title=title)
    fig.update_layout(
        title={"x": 0.02},
        margin={"l": 60, "r": 20, "t": 60, "b": 60},
        legend_title_text="",
    )
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    return fig

def time_line(df, x, y, color=None, title: str = ""):
    fig = px.line(df, x=x, y=y, color=color, title=title)
    fig.update_layout(title={"x": 0.02})
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    return fig

def facet_hist(df, x, facet_col, title: str = ""):
    fig = px.histogram(df, x=x, facet_col=facet_col, title=title)
    fig.update_layout(title={"x": 0.02})
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text="count")
    return fig
```

### Visualization pitfalls + checks

* Don’t let default category ordering mislead → set order via sorting before plotting.
* Avoid overplotting: use transparency (later) or aggregate first.
* Facets can hide scale differences; decide whether to share axes intentionally.
* Exported figures must have readable labels; check outside notebook.

---

# 16) ETL pipeline patterns (job-ready)

**What it’s for:** Produce clean, reproducible datasets with logging, validations, and clean outputs.

## ETL definition

* **Extract**: read from `data/raw` or `data/cache` (offline-first)
* **Transform**: pure functions and deterministic steps
* **Load**: write to `data/processed` as Parquet/CSV (and later optionally load to PostgreSQL) + minimal metadata

### Canonical operations / patterns

* A single `run_etl()` that:

  * logs inputs, row counts, output paths
  * validates invariants
  * writes outputs idempotently
* Use configs for paths and parameters (dates, filters, version tags).
* Write a run summary JSON next to outputs.

### Pitfalls + checks

* Non-idempotent outputs (appending each run) → duplicates.
* Silent schema drift across runs → broken notebooks.
* Writing CSV with implicit types → lost dtypes; prefer Parquet for processed.

## Minimal ETL module template

```python
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging
import pandas as pd

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class ETLConfig:
    root: Path
    raw_orders: Path
    raw_users: Path
    out_orders_enriched: Path
    run_meta: Path

def load_inputs(cfg: ETLConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = pd.read_csv(cfg.raw_orders, dtype={"order_id":"string","user_id":"string"})
    users = pd.read_csv(cfg.raw_users, dtype={"user_id":"string"})
    return orders, users

def transform(orders: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    require_columns(orders, ["order_id","user_id","amount","created_at","status"])
    require_columns(users, ["user_id"])
    assert_unique_key(users, "user_id")

    orders2 = (
        orders.pipe(enforce_schema)
              .pipe(parse_datetime, col="created_at", utc=True)
              .assign(status=lambda d: normalize_text(d["status"]))
    )

    out = safe_left_join(orders2, users, on="user_id", validate="many_to_one", suffixes=("", "_user"))
    assert len(out) == len(orders2), "Row count changed on left join"
    return out

def load_outputs(df: pd.DataFrame, cfg: ETLConfig) -> None:
    cfg.out_orders_enriched.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.out_orders_enriched, index=False)

def write_run_meta(cfg: ETLConfig, *, rows_out: int) -> None:
    cfg.run_meta.parent.mkdir(parents=True, exist_ok=True)
    meta = {"config": {k: str(v) for k, v in asdict(cfg).items()}, "rows_out": rows_out}
    cfg.run_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def run_etl(cfg: ETLConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log.info("Loading inputs")
    orders, users = load_inputs(cfg)

    log.info("Transforming (orders=%s, users=%s)", len(orders), len(users))
    out = transform(orders, users)

    log.info("Writing output: %s (%s rows)", cfg.out_orders_enriched, len(out))
    load_outputs(out, cfg)
    write_run_meta(cfg, rows_out=len(out))
```

## “Run it” entrypoint template (optional)

```python
from pathlib import Path
from project_name.etl import ETLConfig, run_etl

ROOT = Path(__file__).resolve().parents[1]

cfg = ETLConfig(
    root=ROOT,
    raw_orders=ROOT / "data" / "raw" / "orders.csv",
    raw_users=ROOT / "data" / "raw" / "users.csv",
    out_orders_enriched=ROOT / "data" / "processed" / "orders_enriched.parquet",
    run_meta=ROOT / "data" / "processed" / "_run_meta.json",
)

run_etl(cfg)
```

---

# 17) Outputs, metadata, and handoff quality

**What it’s for:** Make your processed data and findings easy for others (and future you) to use correctly.

### Canonical operations / patterns

* Prefer writing **Parquet** to `data/processed/`.
* Also write a small **data dictionary** or schema summary (optional) and **run metadata**.
* Export figures to `reports/figures/` with stable filenames.
* Keep a short `reports/summary.md` with:

  * key findings
  * definitions
  * caveats / limitations
  * next questions

### Pitfalls + checks

* Processed data without documented meaning → misinterpretation.
* No run metadata → can’t reproduce numbers.
* Mixing “analysis-only columns” into processed tables → confusion.

### Minimal metadata template

```python
import pandas as pd

def schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "n_missing": [int(df[c].isna().sum()) for c in df.columns],
    }).sort_values("n_missing", ascending=False)
```

---

# 18) SQL for analytics — DuckDB (local) + PostgreSQL (production baseline)

**What it’s for:** Quick aggregations and joins directly on CSV/Parquet when SQL is faster to express than pandas. In industry this often runs in PostgreSQL; we use DuckDB here because it’s zero-setup and offline-first.

### Canonical operations / patterns

* Use DuckDB to query local files without setting up a database.
* Treat SQL as a **view layer** for analytics, not the core transform engine (unless you choose it).

### Pitfalls + checks

* Don’t maintain two separate sources of truth (SQL transforms + pandas transforms) unless clearly separated.
* Always validate row counts and key uniqueness similarly.

### Minimal template

```python
import duckdb
from pathlib import Path

def query_parquet(path: Path, sql: str):
    con = duckdb.connect()
    con.execute("CREATE VIEW t AS SELECT * FROM read_parquet(?)", [str(path)])
    return con.execute(sql).df()

# Example:
# df = query_parquet(processed_path, "SELECT country, COUNT(*) n, SUM(amount) revenue FROM t GROUP BY 1 ORDER BY revenue DESC")
```

---

# Techniques appendix / index (mini internal wiki for later notebooks)

Create mini-notebooks later (optional deep dives), each tightly scoped:

* **01_missingness_patterns.ipynb**

  * missingness heatmaps/tables, “missing not at random” intuition, flag vs impute patterns
* **02_join_validation.ipynb**

  * uniqueness checks, `merge(validate=...)`, join explosion detection, match-rate reporting
* **03_datetime_recipes.ipynb**

  * parsing formats, timezones, resampling, rolling windows, cohort definitions
* **04_outliers_and_robust_stats.ipynb**

  * IQR rules, winsorization for viz, median/IQR summaries, effect on conclusions
* **05_tidy_data_and_reshape.ipynb**

  * melt/pivot patterns, wide vs long, “one observation per row” exercises
* **06_bootstrap_uncertainty.ipynb**

  * bootstrap CI for mean/median/rates, comparing groups, interpreting variability
* **07_chart_choice_gallery.ipynb**

  * same dataset → different charts; when each clarifies vs misleads
* **08_reporting_handoff_checklist.ipynb**

  * exporting figures, run metadata, summary writing templates

---

# Deliverables

## 1) Reproducible ETL

* [ ] A script/module (e.g., `src/project_name/etl.py`) that runs end-to-end:

  * Extract (from `data/raw` and/or cached API responses)
  * Transform (pure functions, `.pipe`, deterministic)
  * Load (`data/processed/*.parquet` and/or `.csv`)
* [ ] Includes:

  * logging (row counts + file paths)
  * validations (required columns, uniqueness, join validation, basic ranges)
  * idempotent outputs (safe to rerun)
  * run metadata JSON (config + counts)

## 2) EDA report notebook

* [ ] Notebook reads **only from `data/processed/`** (not raw).
* [ ] Answers 3–6 concrete questions with:

  * summary tables
  * Plotly charts (titled, labeled)
  * interpretations with caveats (sample sizes, missingness, bias risks)
* [ ] Exports key figures to `reports/figures/`.

## 3) Short written summary

* [ ] `reports/summary.md` (or similar) containing:

  * key findings (bulleted, quantified)
  * definitions (metrics and filters used)
  * data quality caveats (missingness, duplicates, join coverage, outliers)
  * recommended next steps / questions
