from nbhelper import md, code, save

md(r"""
# 03 · Large Data on Limited RAM

Sooner or later a dataset will not fit comfortably in memory: a laptop has 8 to 16 GB, a small cloud
server 2 to 4 GB, and pandas often needs several times the file size while loading. This module shows how
to measure memory honestly and the techniques, and the alternative libraries, that keep it under control.

**You will learn**
1. How to measure memory: object sizes, and the **peak** memory of a whole process
2. Shrinking a DataFrame: numeric downcasting and categoricals
3. Streaming: process data in chunks or with generators instead of loading everything
4. Columnar files: read only the columns and rows you need
5. Alternatives to pandas: **Polars**, **DuckDB**, **PyArrow** (and when Dask or Spark make sense)
6. A benchmark: the same query six ways, timed and with peak memory measured
""")

md(r"""
## 1. Measuring memory
`sys.getsizeof` gives the size of one Python object (not what it contains); `DataFrame.memory_usage(deep=True)`
gives the real size of a table, strings included; `psutil` reports what the whole process uses.
""")
code(r"""
import sys, os, psutil
import numpy as np
import pandas as pd

proc = psutil.Process(os.getpid())
print(f"this kernel uses {proc.memory_info().rss / 1e6:.0f} MB right now")

print(sys.getsizeof(1), "bytes for an int,", sys.getsizeof(1.0), "for a float,", sys.getsizeof("a"), "for a 1-character str")
print(f"a NumPy array of 1M float64: {np.zeros(1_000_000).nbytes / 1e6:.0f} MB; a list of 1M floats: "
      f"~{(sys.getsizeof([0.0] * 1_000_000) + 1_000_000 * sys.getsizeof(0.0)) / 1e6:.0f} MB")
""")

md(r"""
## 2. Shrinking a DataFrame
pandas defaults to 64-bit numbers and Python strings. Most columns need far less.
""")
code(r"""
rng = np.random.default_rng(0)
n = 1_000_000
df = pd.DataFrame({
    "sensor_id": rng.integers(1, 500, n),                    # fits in int16
    "temperature": rng.normal(21, 4, n),                     # float32 is plenty for a sensor
    "humidity": rng.integers(0, 101, n),                     # fits in uint8
    "status": rng.choice(["ok", "warning", "fault"], n, p=[0.9, 0.08, 0.02]),  # 3 distinct values
    "site": rng.choice([f"site_{i}" for i in range(40)], n),
})

def mb(frame):
    return frame.memory_usage(deep=True).sum() / 1e6

before = mb(df)
small = df.assign(
    sensor_id=pd.to_numeric(df["sensor_id"], downcast="integer"),
    temperature=df["temperature"].astype("float32"),
    humidity=pd.to_numeric(df["humidity"], downcast="unsigned"),
    status=df["status"].astype("category"),
    site=df["site"].astype("category"),
)
print(df.dtypes.to_dict()); print(small.dtypes.to_dict())
print(f"before {before:.0f} MB -> after {mb(small):.0f} MB ({before / mb(small):.1f}x smaller)")
""")

md(r"""
A **category** stores each distinct string once and keeps small integer codes per row. It is the single
biggest win for text columns with few distinct values, and groupby on it is faster too:
""")
code(r"""
import timeit
t_obj = timeit.timeit(lambda: df.groupby("site")["temperature"].mean(), number=5) / 5
t_cat = timeit.timeit(lambda: small.groupby("site", observed=True)["temperature"].mean(), number=5) / 5
print(f"groupby on text: {t_obj * 1000:.0f} ms, on category: {t_cat * 1000:.0f} ms")
""")

md(r"""
## 3. Streaming: chunks and generators
If the result is small (a sum, a count, an average) there is no need to hold all rows at once.
""")
code(r"""
import tempfile
tmp = tempfile.mkdtemp()

# a 3-million-row dataset, written once as CSV and as Parquet, used for the rest of the module
N = 3_000_000
big = pd.DataFrame({
    "sensor_id": rng.integers(1, 500, N).astype("int16"),
    "temperature": rng.normal(21, 4, N).astype("float32"),
    "status": pd.Categorical(rng.choice(["ok", "warning", "fault"], N, p=[0.9, 0.08, 0.02])),
    "site": pd.Categorical(rng.choice([f"site_{i}" for i in range(40)], N)),
    "note": rng.choice(["", "manual check", "recalibrated", "battery low"], N),
})
CSV, PARQUET = os.path.join(tmp, "big.csv"), os.path.join(tmp, "big.parquet")
big.to_csv(CSV, index=False)
big.to_parquet(PARQUET, index=False, row_group_size=250_000)
print(f"CSV {os.path.getsize(CSV) / 1e6:.0f} MB, Parquet {os.path.getsize(PARQUET) / 1e6:.0f} MB")
del big
""")
code(r"""
# mean temperature of 'fault' readings per site, reading the CSV 250,000 rows at a time
sums, counts = {}, {}
for chunk in pd.read_csv(CSV, chunksize=250_000, usecols=["site", "status", "temperature"]):
    faults = chunk[chunk["status"] == "fault"]
    g = faults.groupby("site")["temperature"]
    for site, s in g.sum().items():
        sums[site] = sums.get(site, 0.0) + s
    for site, c in g.count().items():
        counts[site] = counts.get(site, 0) + c
streamed = pd.Series({k: sums[k] / counts[k] for k in sums}).sort_index()
print(streamed.head())
""")
code(r"""
# the same idea with a plain generator: a pipeline that never holds the file in memory
def read_lines(path):
    with open(path) as f:
        next(f)                                   # skip the header
        for line in f:
            yield line.rstrip("\n").split(",")

def faults_only(rows):
    return (r for r in rows if r[2] == "fault")   # columns: sensor_id, temperature, status, site, note

count = sum(1 for _ in faults_only(read_lines(CSV)))
print(f"fault rows: {count:,}")
""")

md(r"""
## 4. Columnar files: read only what you need
Parquet stores each column separately and keeps min/max statistics per block of rows (a *row group*).
A reader can therefore skip columns it does not need (**column pruning**) and whole row groups that
cannot match a filter (**predicate pushdown**).
""")
code(r"""
import pyarrow.parquet as pq
meta = pq.ParquetFile(PARQUET).metadata
print(f"{meta.num_rows:,} rows in {meta.num_row_groups} row groups, {meta.num_columns} columns")

subset = pd.read_parquet(PARQUET, columns=["site", "temperature"], filters=[("status", "==", "fault")])
print(subset.shape, f"{mb(subset):.1f} MB loaded instead of the whole table")
""")

md(r"""
## 5. Alternatives to pandas

| Library | What it is | Strengths |
|---|---|---|
| **Polars** | DataFrame library written in Rust, built on Apache Arrow | multi-threaded, lazy queries with an optimiser, streaming for larger-than-RAM data |
| **DuckDB** | an in-process analytical SQL database | SQL directly on Parquet/CSV files, out-of-core execution, very low memory |
| **PyArrow** | the Python library for Apache Arrow, the columnar memory format the others share | zero-copy exchange between tools, datasets, compute functions |
| **Dask** | pandas-like API split into partitions | scales pandas code to many cores or a cluster |
| **Spark (PySpark)** | distributed engine | data that needs a cluster (hundreds of GB and more) |

Rule of thumb: up to a few GB, pandas with good types is fine; beyond that, or when memory is tight,
Polars or DuckDB on Parquet usually solve it on one machine; reach for Dask or Spark only when one
machine is truly not enough.
""")
code(r"""
import polars as pl, duckdb

print(pl.scan_parquet(PARQUET)                 # lazy: nothing is read yet
        .filter(pl.col("status") == "fault")
        .group_by("site").agg(pl.col("temperature").mean())
        .sort("site").head(3).collect())

print(duckdb.sql(f"select site, avg(temperature) as t from '{PARQUET}' where status = 'fault' "
                 "group by site order by site limit 3").df())
""")

md(r"""
## 6. Benchmark: one query, six ways
Each approach runs in a **fresh Python process** so its peak memory (the operating system's maximum
resident set size) is measured on its own, without anything this notebook already holds. Native
libraries allocate memory outside Python, so this is the only fair way to compare them.
""")
code(r'''
import subprocess, json, textwrap

QUERY = "mean temperature of fault readings per site"
approaches = {
    "pandas read_csv (all columns)": f"""
df = pd.read_csv({CSV!r})
r = df[df.status == 'fault'].groupby('site').temperature.mean()""",
    "pandas read_csv (3 columns, typed)": f"""
df = pd.read_csv({CSV!r}, usecols=['site', 'status', 'temperature'], dtype={{'site': 'category', 'status': 'category', 'temperature': 'float32'}})
r = df[df.status == 'fault'].groupby('site', observed=True).temperature.mean()""",
    "pandas read_csv in chunks": f"""
parts = [c[c.status == 'fault'] for c in pd.read_csv({CSV!r}, usecols=['site', 'status', 'temperature'], chunksize=250_000)]
r = pd.concat(parts).groupby('site').temperature.mean()""",
    "pandas read_parquet (pruned + filtered)": f"""
df = pd.read_parquet({PARQUET!r}, columns=['site', 'temperature'], filters=[('status', '==', 'fault')])
r = df.groupby('site', observed=True).temperature.mean()""",
    "Polars lazy scan": f"""
import polars as pl
r = pl.scan_parquet({PARQUET!r}).filter(pl.col('status') == 'fault').group_by('site').agg(pl.col('temperature').mean()).collect()""",
    "DuckDB SQL": f"""
import duckdb
r = duckdb.sql("select site, avg(temperature) from '{PARQUET}' where status = 'fault' group by site").df()""",
}

def run(snippet):
    # Peak memory = VmHWM ("high water mark") from /proc/self/status, in kB. Unlike getrusage's
    # ru_maxrss it belongs to the new program only: Linux carries ru_maxrss over from the parent
    # (this notebook) through fork and exec, which would hide every difference. (Linux only.)
    prog = ("import time, pandas as pd\nt0 = time.perf_counter()\n" + textwrap.dedent(snippet) +
            "\nhwm = next(l for l in open('/proc/self/status') if l.startswith('VmHWM')).split()[1]"
            "\nprint(time.perf_counter() - t0, hwm)")
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, check=True).stdout.split()
    return float(out[-2]), int(out[-1]) / 1024          # seconds, MB

baseline_mb = run("pass")[1]                            # what an empty python + pandas import costs
bench = pd.DataFrame([dict(approach=k, seconds=s, peak_mb=m - baseline_mb) for k, (s, m) in
                      ((k, run(v)) for k, v in approaches.items())]).set_index("approach").round(2)
print(f"{QUERY}; peak memory above a bare 'import pandas' process ({baseline_mb:.0f} MB)")
bench
''')
code(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
bench["seconds"].plot.barh(ax=ax[0], color="#0ea5e9"); ax[0].set_xlabel("time (s)")
bench["peak_mb"].plot.barh(ax=ax[1], color="#4f46e5"); ax[1].set_xlabel("extra peak memory (MB)")
ax[0].invert_yaxis(); plt.tight_layout(); plt.show()
""")

md(r"""
**What the numbers say.** Two different goals, two different winners:
- **Speed** comes from the file format and the engine. Reading a columnar Parquet file with column pruning
  and a filter, or letting DuckDB or Polars run the query, is roughly 6 to 20 times faster than parsing the
  CSV, because nothing irrelevant is ever read or converted from text.
- **Memory** comes from holding less. Loading the whole CSV with default types is by far the most
  memory-hungry option. Asking pandas for only three columns with compact types, or streaming in chunks,
  cuts the peak about five- to six-fold, but parsing text stays slow.
- The extra memory shown for Parquet, Polars and DuckDB is mostly the libraries themselves being loaded
  (the Arrow and DuckDB engines are tens of MB each); at this data size that fixed cost dominates. It stays
  roughly constant as data grows, while a full pandas load grows with the file.

In practice: store data as Parquet, select columns and types up front, and let a query engine (DuckDB or
Polars) do the filtering and aggregation; stream when even the selected data does not fit.

## Checklist for memory-constrained processing
1. **Measure first**: `memory_usage(deep=True)`, `psutil`, or the peak of a test run.
2. **Store as Parquet**, not CSV, and read only the needed columns and rows.
3. **Use the right types**: `category` for repeated strings, `float32`/small integers where precision allows.
4. **Stream** when the result is small: chunks, generators, or a lazy engine.
5. **Delete what you no longer need** (`del df`) and avoid accidental copies.
6. **Let a query engine do the work** (DuckDB, Polars) when pandas would load everything.
7. Only then think about more RAM or a cluster.

## Exercises
1. Repeat the benchmark with `sensor_id` as the group key. Does the ranking change?
2. Add an approach using `pyarrow.dataset` with a filter expression and compare it.
3. Use `pl.scan_csv` instead of `scan_parquet`: how much of Polars' advantage comes from the file format?
""")

save("03_large_data_memory.ipynb")
