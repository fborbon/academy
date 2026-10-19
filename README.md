# Academy

Free, hands-on courses, listed on the Academy home page at [academy.forwardforecasting.eu](https://academy.forwardforecasting.eu/) (Spanish at [/es/](https://academy.forwardforecasting.eu/es/)). Every course is a set of runnable Jupyter notebooks with their results saved, readable on the course website and runnable on an ordinary laptop CPU, with no GPU and no API keys.

| Course | Folder | Website |
|---|---|---|
| **Neural Networks 1001**: from a single neuron to a GPT in nine notebooks, plus a capstone on a real industrial defect-inspection case | [`neural-networks/`](neural-networks/) | [neural-networks.academy.forwardforecasting.eu](https://neural-networks.academy.forwardforecasting.eu/) |
| **Redes Neuronales 1001**: the same course in Spanish | [`neural-networks/es/`](neural-networks/es/) | [neural-networks.academy.forwardforecasting.eu/es/](https://neural-networks.academy.forwardforecasting.eu/es/) |
| **Python 1001**: Intro, Medium and Advanced modules, from data structures to frontier science, forecasting and a Kaggle capstone, in thirteen notebooks | [`python/`](python/) | [python.academy.forwardforecasting.eu](https://python.academy.forwardforecasting.eu/) |
| **Python 1001 (español)**: the same course in Spanish | [`python/es/`](python/es/) | [python.academy.forwardforecasting.eu/es/](https://python.academy.forwardforecasting.eu/es/) |

Each course folder has its own README with the syllabus, how to run it, and how its website is built.

## Academy home page

[`home/`](home/) builds the course catalogue at academy.forwardforecasting.eu, in English (site root) and Spanish (`/es/`). Course names, descriptions, parts, notebook counts and study hours are read from each course's `site/build_site.py`, so the catalogue never drifts from the course websites; only the catalogue's own text (levels, prerequisites, suggested path) lives in `home/build_site.py`. Each course home page has a button back to it.

```
python3 home/build_site.py     # writes home/dist/ (needs nbconvert, which the course builders import)
```

Pushes that touch `home/` or either course's `site/build_site.py` run `.github/workflows/deploy-academy-home.yml`, which copies `home/dist/` into the portfolio repository under `academy/home/`; its deploy publishes it. The nginx site is versioned in [`infra/nginx/academy-home.conf`](infra/nginx/academy-home.conf).

| Folder | Contents |
|---|---|
| `home/` | catalogue builder and templates (page, stylesheet, theme toggle) |
| `neural-networks/`, `python/` | one folder per course: notebooks, builders, translations, course website |
| `infra/nginx/` | the web server configuration of each site |

## Contact

Questions or suggestions: forwardforecasting@gmail.com
