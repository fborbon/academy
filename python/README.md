# Python 1001

A free, hands-on Python course in three modules: **Intro**, **Medium** and **Advanced**. Thirteen Jupyter
notebooks with their results saved, so the whole course can be read on the website, and every notebook runs
on an ordinary laptop CPU in a few minutes.

**Website:** [www.forwardforecasting.eu/academy/python/](https://www.forwardforecasting.eu/academy/python/)
(its final address, python.academy.forwardforecasting.eu, goes live once its TLS certificate is issued).

## Table of contents
1. [Syllabus](#syllabus)
2. [Run it yourself](#run-it-yourself)
3. [Libraries used](#libraries-used)
4. [Data and licences](#data-and-licences)
5. [How the notebooks and the website are built](#how-the-notebooks-and-the-website-are-built)
6. [Cost and resources](#cost-and-resources)

## Syllabus

| # | Notebook | Module | What it covers |
|---|---|---|---|
| 01 | Python Data Structures | Intro | lists, tuples, dicts, sets, `collections`, functions, a first look at classes, comprehensions, generators, exceptions, lookup speed measured |
| 02 | Data Formats and pandas | Intro | the DataFrame; CSV, JSON, Excel, Parquet, Feather, SQLite, YAML, TOML and pickle compared on the same data (size, speed, types kept); pitfalls |
| 03 | Large Data on Limited RAM | Intro | measuring memory, compact dtypes and categoricals, chunking and generators, Parquet pruning, Polars, DuckDB and PyArrow benchmarked per process; **PySpark**: the same queries written with the DataFrame API (`F.col`, `groupBy`, window functions) and with SQL (`spark.sql(...)`), query plans, partitioned writes |
| 04 | Environments, Packages, Containers and Editors | Intro | `venv` (demonstrated), pip, uv, pipx, conda, apt, snap, brew; lock files; Docker for deployment; VS Code, PyCharm, Eclipse + PyDev, Vim, JupyterLab, Spyder; a project layout |
| 05 | Object-Oriented Python | Medium | the four pillars, special methods, properties, class and static methods, dataclasses, ABCs vs. Protocols, composition, MRO, `__slots__` |
| 06 | Pythonic Techniques | Medium | iterators, `itertools`, decorators, context managers, functional tools, type hints, logging, pytest, threads vs. processes vs. asyncio (measured) |
| 07 | Libraries: Speed Up, Package, Publish, Contribute | Medium | NumPy, Numba, Cython and C via ctypes benchmarked; `pyproject.toml`, building and checking a wheel, installing it in a clean environment; trusted publishing to PyPI; SemVer, specifiers, deprecations; contributing upstream |
| 08 | Life Sciences: Biology and Medicine | Advanced | Biopython sequences, simulated evolution, alignment and a neighbour-joining tree; Kaplan-Meier, log-rank and Cox regression on the GBSG2 breast-cancer trial; a diagnostic classifier tuned for sensitivity |
| 09 | Physics and Space Research | Advanced | SymPy derivation of the pendulum, SciPy simulation, FFT, Astropy units and constants, Kepler's third law on the planets, an orbit validated by conservation laws, sky coordinates |
| 10 | Economics and Econometrics | Advanced | statsmodels on US macro data: Okun's law with HAC errors and its stability by decade; ARIMA forecasts in a rolling backtest against naive baselines |
| 11 | AI Frameworks: PyTorch and TensorFlow | Advanced | the same CNN in both frameworks: autograd vs. GradientTape, explicit loop vs. `fit`, saving and reloading, how to choose |
| 12 | Time Series Forecasting | Advanced | seasonal naive, SARIMA, Prophet, XGBoost, GRU (PyTorch), LSTM (Keras) and a Transformer on two problems: hourly bike demand (tabular data with strong features, where gradient boosting wins) and river flow driven by rain (hidden long-memory state, where recurrent networks win), and why |
| 13 | Capstone: Solving a Kaggle Challenge | Advanced | House Prices end to end: data licensing, log-RMSE metric, exploration, feature engineering in a Pipeline, cross-validation, blending (held-out RMSE(log) 0.110), submission file |

The material of the original interview-refresher notebook (core types, functions, classes and the four OOP
pillars, comprehensions, exceptions, pandas basics) is folded into notebooks 01, 02 and 05.

## Run it yourself

```bash
git clone https://github.com/fborbon/academy.git
cd academy/python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
jupyter lab
```

Or open any notebook in Google Colab from its lesson page. Notebook 07 compiles C and Cython code and
needs a C compiler (`gcc` on Linux, Xcode command line tools on macOS, Build Tools on Windows; Colab has
one). Notebook 03 runs PySpark, which needs a Java runtime 17 or newer (`apt install openjdk-17-jre-headless`; Colab has one). Notebooks 12 and 13 download small datasets (about 1 MB each) from OpenML on first run.

## Libraries used

| Library | Used for | Notebooks |
|---|---|---|
| pandas, NumPy | tables and arrays throughout | all |
| PyArrow, Polars, DuckDB | columnar files and out-of-core queries | 02, 03 |
| psutil | process memory | 03 |
| pytest | testing | 06 |
| Numba, Cython, build, twine, packaging | speeding up code, packaging and publishing | 07 |
| Biopython | sequences, alignment, phylogenetics | 08 |
| lifelines | survival analysis | 08 |
| scikit-learn | classifiers, pipelines, cross-validation | 08, 11, 13 |
| SymPy, SciPy | symbolic maths, ODEs, signal processing | 09 |
| Astropy | units, constants, sky coordinates | 09 |
| statsmodels | regression, time series | 10, 12 |
| PySpark | distributed DataFrames and SQL | 03 |
| XGBoost, Prophet | gradient-boosted trees, decomposable forecasting | 12 |
| PyTorch, TensorFlow/Keras | deep learning frameworks | 11, 12 |
| matplotlib | every figure | all |

## Data and licences

Most data is generated inside the notebooks or ships with the libraries (scikit-learn, statsmodels,
lifelines). Notebook 12 uses the UCI Bike Sharing dataset (Fanaee-T and Gama, 2013, licence CC BY 4.0) from
OpenML (dataset 42712). The Kaggle capstone deliberately does **not** use Kaggle's competition files, whose rules
restrict use and redistribution: it uses the original **Ames Housing** dataset compiled for teaching by
Dean De Cock (*Journal of Statistics Education*, 2011), from OpenML (dataset 43926, licence: public), and
explains how to download the competition files with your own account to submit. Credits are listed on the
course home page. Code and text: MIT licence.

## How the notebooks and the website are built

The same pipeline as the other academy course:

- `builders/build_NN.py` writes each notebook from Python (`builders/nbhelper.py`); never edit a notebook
  by hand. `builders/rebuild.py NN --execute` regenerates and runs it (one at a time), and
  `rebuild.py all --check` verifies in CI that every committed notebook matches its builder.
- `site/build_site.py` renders the saved outputs with nbconvert into a static site (`site/dist/`), with a
  syllabus, lesson pages, progress kept in the browser, and downloadable notebooks.
- `.github/workflows/deploy-python.yml` (repository root) checks, builds and copies the site into the
  portfolio site repository, whose own deploy publishes it.
- The nginx virtual host for the subdomain is versioned in `infra/nginx/academy-python.conf`.

## Cost and resources

A static site: **$0** extra. It is served by the existing web server from files (no process, no memory),
the build runs on free GitHub Actions minutes, and nothing in the course calls a paid API. Executing all
thirteen notebooks locally takes about 15 minutes on a laptop CPU.

## Contact

Questions or suggestions: forwardforecasting@gmail.com
