# Day 4: EDA and Visualization with Plotly

## Learning Objectives

By the end of Day 4, you will be able to:
- Create clear, publication-ready charts with Plotly
- Export figures to files
- Compute bootstrap confidence intervals
- Answer business questions with data

## Activities

### Task 1: Create Visualization Helpers (20 minutes)

Create `src/data_workflow/viz.py` with:

1. **`bar_sorted(df, x, y, title)`** - Sorted bar chart
2. **`time_line(df, x, y, title)`** - Line chart for trends
3. **`histogram_chart(df, x, nbins, title)`** - Distribution histogram
4. **`save_fig(fig, path, scale=2)`** - Export figure to PNG

**References:**
- [plotly.express documentation](https://plotly.com/python/plotly-express/)
- [plotly figure.update_layout](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.update_layout)
- [plotly figure.write_image](https://plotly.com/python/static-image-export/)

**Key Concepts:**
- Use `plotly.express` for quick charts
- Customize with `fig.update_layout()` and `fig.update_xaxes/yaxes()`
- Export requires `kaleido` package: `pip install kaleido`

### Task 2: Create Bootstrap Function (15 minutes)

Create `src/data_workflow/utils.py` with:

**`bootstrap_diff_means(a, b, n_boot=2000, seed=0)`**
- Resample each group with replacement
- Compute difference in means for each resample
- Return observed difference and 95% confidence interval

**References:**
- [numpy.random.default_rng](https://numpy.org/doc/stable/reference/random/generator.html)
- [numpy.quantile](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)

### Task 3: Create EDA Notebook (60 minutes)

Create `notebooks/eda.ipynb` that:
1. Loads `analytics_table.parquet`
2. Performs data audit (rows, dtypes, missingness)
3. Answers 3-6 questions with:
   - Summary tables
   - Visualizations
   - Interpretations and caveats
4. Includes one bootstrap comparison
5. Exports at least 3 figures to `reports/figures/`


## Progressive Hints

### If charts don't export:

**Hint 1:** Install kaleido: `pip install kaleido`

**Hint 2:** Make sure `reports/figures/` directory exists before exporting.

### If bootstrap fails:

**Hint 1:** Use fixed seed for reproducibility: `seed=0`

**Hint 2:** Drop missing values before resampling:
```python
a_clean = pd.to_numeric(a, errors="coerce").dropna()
```

## Checklist

- [ ] `viz.py` has visualization functions
- [ ] `utils.py` has bootstrap function
- [ ] `notebooks/eda.ipynb` exists and runs
- [ ] At least 3 figures exported to `reports/figures/`

