# Day 5: ETL Pipeline and Handoff

## Learning Objectives

By the end of Day 5, you will be able to:
- Package your work as a production-ready ETL pipeline
- Generate run metadata for reproducibility
- Write a clear summary with findings and caveats
- Ensure your repo is ready for handoff

## Activities

### Task 1: Create ETL Module (60 minutes)

Create `src/data_workflow/etl.py` with:

1. **`ETLConfig` dataclass** - Configuration with all paths
2. **`load_inputs(cfg)`** - Extract raw data
3. **`transform(orders, users)`** - Clean and enrich data
4. **`load_outputs(analytics, users, cfg)`** - Write processed outputs
5. **`write_run_meta(cfg, analytics)`** - Write run metadata JSON
6. **`run_etl(cfg)`** - Orchestrate the pipeline

**References:**
- [Python logging module](https://docs.python.org/3/library/logging.html)
- [dataclasses.asdict](https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict)
- [JSON module](https://docs.python.org/3/library/json.html)

### Task 2: Create Main ETL Script (15 minutes)

Create `scripts/run_etl.py` that:
- Defines `ROOT` path
- Creates `ETLConfig` with all paths
- Calls `run_etl(cfg)`

### Task 3: Write Summary (25 minutes)

Create `reports/summary.md` with:
- **Key Findings**: Bulleted, quantified results
- **Definitions**: Metrics and filters used
- **Data Quality Caveats**: Missingness, duplicates, join coverage, outliers
- **Next Questions**: Recommended follow-up analyses


## Progressive Hints

### If ETL fails:

**Hint 1:** Make sure all Day 1-3 functions are working first.

**Hint 2:** Add logging to see where it fails:
```python
log.info("Starting transform...")
analytics = transform(orders, users)
log.info("Transform complete: %s rows", len(analytics))
```

### If metadata writing fails:

**Hint 1:** Create parent directory first:
```python
cfg.run_meta.parent.mkdir(parents=True, exist_ok=True)
```

**Hint 2:** Convert Path objects to strings for JSON:
```python
"config": {k: str(v) for k, v in asdict(cfg).items()}
```

## Checklist

- [ ] `etl.py` has complete ETL pipeline
- [ ] `scripts/run_etl.py` runs successfully
- [ ] Creates `data/processed/_run_meta.json`
- [ ] `reports/summary.md` has all required sections

