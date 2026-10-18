"""Build the static course website for Python 1001 from the executed notebooks.

    python3 site/build_site.py            # writes site/dist/

Two editions are built from LANGS: English at the site root and Spanish under /es/ (notebooks from
python/es/). The notebooks are rendered with nbconvert's "basic" template
(body only) and wrapped in the course layout from site/templates. Nothing is fetched at build time.
"""
import html
import json
import os
import re
import shutil
from datetime import date
from string import Template

import nbformat
from nbconvert import HTMLExporter
from pygments.formatters import HtmlFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIST = os.path.join(HERE, "dist")
REPO = "https://github.com/fborbon/academy"
REPO_PATH = "python"
SITE_URL = "https://python.academy.forwardforecasting.eu"
FIELD_NOTES = "https://education.forwardforecasting.eu/"

# per-module data that does not depend on the language
BASE = [
    dict(n="01", track=0, minutes=60, cpu="5 s"),
    dict(n="02", track=0, minutes=60, cpu="30 s"),
    dict(n="03", track=0, minutes=75, cpu="1 min"),
    dict(n="04", track=0, minutes=60, cpu="15 s"),
    dict(n="05", track=1, minutes=75, cpu="5 s"),
    dict(n="06", track=1, minutes=90, cpu="10 s"),
    dict(n="07", track=1, minutes=90, cpu="1 min"),
    dict(n="08", track=2, minutes=75, cpu="20 s"),
    dict(n="09", track=2, minutes=75, cpu="20 s"),
    dict(n="10", track=2, minutes=60, cpu="40 s"),
    dict(n="11", track=2, minutes=60, cpu="30 s"),
    dict(n="12", track=2, minutes=120, cpu="25 min"),
    dict(n="13", track=2, minutes=120, cpu="1 min"),
]

LANGS = {
    "en": dict(
        dir="", nb_dir="", course="Python 1001", course_html="Python <b>1001</b>",
        description="A free, hands-on Python course in three levels: data structures and tooling, professional programming and libraries, and frontier science, AI and forecasting, ending with a Kaggle challenge.",
        switch_label="Español", switch_title="Leer el curso en español",
        tracks=[("Intro", "Data structures, formats, data larger than memory, environments, containers and editors."),
                ("Medium", "Object-oriented and Pythonic programming, testing, concurrency, and building, publishing and maintaining libraries."),
                ("Advanced", "Python at the frontier: biology, medicine, physics, space, economics, AI frameworks, time series forecasting and a Kaggle capstone.")],
        modules=[
            ("01_data_structures.ipynb", "01-data-structures", "Python Data Structures",
             "Lists, tuples, dictionaries and sets, the collections module, functions, a first look at classes, comprehensions, generators and exceptions, and why the choice of structure matters, measured.",
             ["list, tuple, dict, set", "collections", "functions", "generators", "exceptions", "Big-O"]),
            ("02_data_formats.ipynb", "02-data-formats", "Data Formats and pandas",
             "The pandas DataFrame, then CSV, JSON, Excel, Parquet, Feather, SQLite, YAML, TOML and pickle compared on the same data: size, speed and which types survive.",
             ["pandas", "CSV & JSON", "Parquet", "SQLite", "YAML & TOML", "pickle safety"]),
            ("03_large_data_memory.ipynb", "03-large-data-memory", "Large Data on Limited RAM",
             "Measure memory honestly, shrink DataFrames with the right types, stream in chunks, read only what you need from Parquet, and compare pandas with Polars, DuckDB and PyArrow.",
             ["memory profiling", "categoricals", "chunking", "Polars", "DuckDB", "PyArrow"]),
            ("04_environments_and_tools.ipynb", "04-environments-and-tools", "Environments, Packages, Containers and Editors",
             "Isolated environments, pip, uv, pipx, conda, apt and snap, lock files, Docker for deployment, and the editors professionals use: VS Code, PyCharm, Eclipse, Vim.",
             ["venv", "pip & uv", "conda", "lock files", "Docker", "IDEs"]),
            ("05_object_oriented_python.ipynb", "05-object-oriented-python", "Object-Oriented Python",
             "The four pillars, the data model and special methods, properties, dataclasses, abstract base classes and protocols, composition, the MRO and __slots__.",
             ["classes", "dunder methods", "properties", "dataclasses", "Protocol", "composition"]),
            ("06_pythonic_techniques.ipynb", "06-pythonic-techniques", "Pythonic Techniques",
             "Iterators, decorators, context managers, functional tools, type hints, logging, testing with pytest, and threads versus processes versus asyncio, measured.",
             ["generators", "decorators", "context managers", "typing", "pytest", "concurrency"]),
            ("07_libraries_build_publish.ipynb", "07-libraries-build-publish", "Libraries: Speed Up, Package, Publish, Contribute",
             "Speed up hot loops with NumPy, Numba, Cython and C, build a wheel from pyproject.toml, publish with trusted publishing, version without breaking users, and contribute upstream.",
             ["Numba", "Cython", "C extensions", "pyproject.toml", "PyPI", "SemVer & deprecation"]),
            ("08_life_sciences.ipynb", "08-life-sciences", "Life Sciences: Biology and Medicine",
             "Sequences, simulated evolution and a reconstructed family tree with Biopython, survival analysis of a breast-cancer trial with lifelines, and a diagnostic model tuned for sensitivity.",
             ["Biopython", "alignment", "phylogeny", "Kaplan-Meier", "Cox regression", "ROC"]),
            ("09_physics_and_space.ipynb", "09-physics-and-space", "Physics and Space Research",
             "Derive a pendulum symbolically with SymPy, simulate it with SciPy, find its frequency with an FFT, verify Kepler's third law with Astropy, and integrate an orbit checked by conservation laws.",
             ["SymPy", "SciPy ODEs", "FFT", "Astropy units", "Kepler", "orbits"]),
            ("10_economics.ipynb", "10-economics", "Economics and Econometrics",
             "Fifty years of US data with statsmodels: Okun's law with robust standard errors, its stability across decades, and an ARIMA forecast backtested against naive baselines.",
             ["statsmodels", "OLS & HAC", "Okun's law", "ARIMA", "backtesting"]),
            ("11_ai_frameworks.ipynb", "11-ai-frameworks", "AI Frameworks: PyTorch and TensorFlow",
             "The same convolutional network in PyTorch and in TensorFlow/Keras: tensors, automatic differentiation, training loops, saving and exporting, and how to choose.",
             ["PyTorch", "TensorFlow", "Keras", "autograd", "CNN", "ONNX"]),
            ("12_time_series_forecasting.ipynb", "12-time-series-forecasting", "Time Series Forecasting",
             "Seasonal naive, SARIMA, Prophet, XGBoost, a GRU in PyTorch, an LSTM in Keras and a Transformer on two problems: bike demand, where gradient boosting wins, and river flow, where neural networks win, and why.",
             ["seasonal naive", "SARIMA", "Prophet", "XGBoost", "GRU & LSTM", "Transformer"]),
            ("13_kaggle_house_prices.ipynb", "13-kaggle-house-prices", "Capstone: Solving a Kaggle Challenge",
             "Kaggle's House Prices challenge end to end: licensing, the log-RMSE metric, exploration, feature engineering in a Pipeline, cross-validation, blending and a submission file.",
             ["Kaggle", "data licensing", "feature engineering", "Pipeline", "cross-validation", "blending"]),
        ],
        t=dict(
            nav_projects="Projects", toggle_theme="Toggle dark mode",
            footer_data="Data is generated in the notebooks, ships with the libraries, or is downloaded from OpenML with credit. Code and text: MIT licence.",
            completed="✓ completed", minutes="~{m} min", runs="runs in {cpu} on a CPU", part="Module {n}",
            module_of="Notebook {n} of {total}", download="⬇ Download notebook", colab="Open in Colab", github="View on GitHub",
            overview="← Course overview", modules_btn="Notebooks ▾", on_page="On this page", prev="← Previous", next="Next →",
            back="← Back to", finished="Finished? →", overview_short="Course overview", fig_alt="notebook output figure",
            js_mark="Mark notebook {n} as complete", js_done="✓ Notebook {n} completed (click to undo)",
            js_progress="{done} of {total} notebooks completed", js_none="Not started yet",
            js_resume="Continue with notebook {n}: {title} →",
        ),
    ),
    "es": dict(
        dir="es", nb_dir="es", course="Python 1001", course_html="Python <b>1001</b>",
        description="Un curso de Python gratuito y práctico en tres niveles: estructuras de datos y herramientas, programación profesional y bibliotecas, y ciencia de frontera, IA y previsión, con un reto de Kaggle al final.",
        switch_label="English", switch_title="Read the course in English",
        tracks=[("Inicial", "Estructuras de datos, formatos, datos mayores que la memoria, entornos, contenedores y editores."),
                ("Intermedio", "Programación orientada a objetos y pythónica, pruebas, concurrencia, y construir, publicar y mantener bibliotecas."),
                ("Avanzado", "Python en la frontera: biología, medicina, física, espacio, economía, frameworks de IA, previsión de series temporales y un proyecto final de Kaggle.")],
        modules=[
            ("01_estructuras_de_datos.ipynb", "01-estructuras-de-datos", "Estructuras de datos en Python",
             "Listas, tuplas, diccionarios y conjuntos, el módulo collections, funciones, un primer vistazo a las clases, comprensiones, generadores y excepciones, y por qué importa la estructura elegida, medido.",
             ["list, tuple, dict, set", "collections", "funciones", "generadores", "excepciones", "Big-O"]),
            ("02_formatos_de_datos.ipynb", "02-formatos-de-datos", "Formatos de datos y pandas",
             "El DataFrame de pandas y después CSV, JSON, Excel, Parquet, Feather, SQLite, YAML, TOML y pickle comparados con los mismos datos: tamaño, velocidad y qué tipos sobreviven.",
             ["pandas", "CSV y JSON", "Parquet", "SQLite", "YAML y TOML", "seguridad de pickle"]),
            ("03_grandes_datos_y_memoria.ipynb", "03-grandes-datos-y-memoria", "Grandes volúmenes de datos con poca RAM",
             "Mide la memoria con honestidad, reduce DataFrames con los tipos adecuados, procesa por bloques, lee de Parquet solo lo necesario, compara pandas con Polars, DuckDB y PyArrow, y escala con PySpark.",
             ["perfilado de memoria", "categorías", "por bloques", "Polars y DuckDB", "PyArrow", "PySpark"]),
            ("04_entornos_y_herramientas.ipynb", "04-entornos-y-herramientas", "Entornos, paquetes, contenedores y editores",
             "Entornos aislados, pip, uv, pipx, conda, apt y snap, archivos de bloqueo, Docker para desplegar y los editores que usan los profesionales: VS Code, PyCharm, Eclipse, Vim.",
             ["venv", "pip y uv", "conda", "archivos de bloqueo", "Docker", "IDE"]),
            ("05_python_orientado_a_objetos.ipynb", "05-python-orientado-a-objetos", "Python orientado a objetos",
             "Los cuatro pilares, el modelo de datos y los métodos especiales, propiedades, dataclasses, clases base abstractas y protocolos, composición, el MRO y __slots__.",
             ["clases", "métodos dunder", "propiedades", "dataclasses", "Protocol", "composición"]),
            ("06_tecnicas_pythonicas.ipynb", "06-tecnicas-pythonicas", "Técnicas pythónicas",
             "Iteradores, decoradores, gestores de contexto, herramientas funcionales, anotaciones de tipo, logging, pruebas con pytest, e hilos frente a procesos frente a asyncio, medido.",
             ["generadores", "decoradores", "gestores de contexto", "tipado", "pytest", "concurrencia"]),
            ("07_bibliotecas_compilar_publicar.ipynb", "07-bibliotecas-compilar-publicar", "Bibliotecas: acelerar, empaquetar, publicar, contribuir",
             "Acelera bucles críticos con NumPy, Numba, Cython y C, construye un wheel a partir de pyproject.toml, publica con trusted publishing, versiona sin romper a los usuarios y contribuye a otros proyectos.",
             ["Numba", "Cython", "extensiones en C", "pyproject.toml", "PyPI", "SemVer y obsolescencia"]),
            ("08_ciencias_de_la_vida.ipynb", "08-ciencias-de-la-vida", "Ciencias de la vida: biología y medicina",
             "Secuencias, evolución simulada y un árbol genealógico reconstruido con Biopython, análisis de supervivencia de un ensayo de cáncer de mama con lifelines y un modelo de diagnóstico ajustado a la sensibilidad.",
             ["Biopython", "alineamiento", "filogenia", "Kaplan-Meier", "regresión de Cox", "ROC"]),
            ("09_fisica_y_espacio.ipynb", "09-fisica-y-espacio", "Física e investigación espacial",
             "Deduce un péndulo con SymPy, simúlalo con SciPy, encuentra su frecuencia con una FFT, verifica la tercera ley de Kepler con Astropy e integra una órbita comprobada con leyes de conservación.",
             ["SymPy", "EDO con SciPy", "FFT", "unidades con Astropy", "Kepler", "órbitas"]),
            ("10_economia.ipynb", "10-economia", "Economía y econometría",
             "Cincuenta años de datos de EE. UU. con statsmodels: la ley de Okun con errores estándar robustos, su estabilidad por décadas y una previsión ARIMA contrastada frente a referencias ingenuas.",
             ["statsmodels", "MCO y HAC", "ley de Okun", "ARIMA", "backtesting"]),
            ("11_frameworks_de_ia.ipynb", "11-frameworks-de-ia", "Frameworks de IA: PyTorch y TensorFlow",
             "La misma red convolucional en PyTorch y en TensorFlow/Keras: tensores, diferenciación automática, bucles de entrenamiento, guardar y exportar, y cómo elegir.",
             ["PyTorch", "TensorFlow", "Keras", "autograd", "CNN", "ONNX"]),
            ("12_prevision_de_series_temporales.ipynb", "12-prevision-de-series-temporales", "Previsión de series temporales",
             "Ingenuo estacional, SARIMA, Prophet, XGBoost, una GRU en PyTorch, una LSTM en Keras y un Transformer en dos problemas: demanda de bicicletas, donde gana el gradient boosting, y caudal de un río, donde ganan las redes neuronales, y por qué.",
             ["ingenuo estacional", "SARIMA", "Prophet", "XGBoost", "GRU y LSTM", "Transformer"]),
            ("13_kaggle_precios_de_viviendas.ipynb", "13-kaggle-precios-de-viviendas", "Proyecto final: resolver un reto de Kaggle",
             "El reto House Prices de Kaggle de principio a fin: licencias, la métrica RMSE logarítmica, exploración, ingeniería de variables en un Pipeline, validación cruzada, mezcla de modelos y un archivo de envío.",
             ["Kaggle", "licencias de datos", "ingeniería de variables", "Pipeline", "validación cruzada", "mezcla de modelos"]),
        ],
        t=dict(
            nav_projects="Proyectos", toggle_theme="Cambiar a modo oscuro",
            footer_data="Los datos se generan en los notebooks, vienen con las bibliotecas o se descargan de OpenML citando su origen. Código y texto: licencia MIT.",
            completed="✓ completado", minutes="~{m} min", runs="se ejecuta en {cpu} en una CPU", part="Módulo {n}",
            module_of="Notebook {n} de {total}", download="⬇ Descargar notebook", colab="Abrir en Colab", github="Ver en GitHub",
            overview="← Presentación del curso", modules_btn="Notebooks ▾", on_page="En esta página", prev="← Anterior", next="Siguiente →",
            back="← Volver a la", finished="¿Terminaste? →", overview_short="Presentación del curso", fig_alt="figura generada por el notebook",
            js_mark="Marcar el notebook {n} como completado", js_done="✓ Notebook {n} completado (clic para deshacer)",
            js_progress="{done} de {total} notebooks completados", js_none="Todavía sin empezar",
            js_resume="Continúa con el notebook {n}: {title} →",
        ),
    ),
}

FAVICON = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect width='64' height='64' rx='14' fill='%234f46e5'/>"
    "<text x='32' y='42' text-anchor='middle' font-family='monospace' font-size='26' font-weight='700' fill='white'>py</text></svg>"
)

MATHJAX = """
  <script>
    window.MathJax = { tex: { inlineMath: [["$", "$"], ["\\\\(", "\\\\)"]], displayMath: [["$$", "$$"], ["\\\\[", "\\\\]"]], processEscapes: true },
                       options: { skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"] },
                       chtml: { displayAlign: "center" },
                       startup: { pageReady: function () {
                         return MathJax.startup.defaultPageReady().then(function () {
                           document.querySelectorAll("mjx-container").forEach(function (e) { e.classList.add("notranslate"); e.setAttribute("translate", "no"); });
                         }); } } };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>"""


def modules(lang):
    return [dict(b, file=f, slug=s, title=ti, blurb=bl, topics=tp)
            for b, (f, s, ti, bl, tp) in zip(BASE, LANGS[lang]["modules"])]


def read(name, lang="en"):
    localized = os.path.join(HERE, "templates", lang, name)
    path = localized if os.path.exists(localized) else os.path.join(HERE, "templates", name)
    with open(path, encoding="utf-8") as f:
        return f.read()


def url_of(lang, slug=""):
    d = LANGS[lang]["dir"]
    return SITE_URL + "/" + (d + "/" if d else "") + (slug + "/" if slug else "")


def page(lang, title, description, body, prefix, assets, counterpart, switch_href, extra_head=""):
    L, other = LANGS[lang], next((x for x in LANGS if x != lang), lang)
    counterpart = {k: v for k, v in counterpart.items() if k in LANGS}
    alternates = ""
    if len(LANGS) > 1:
        alternates = "".join(f'\n  <link rel="alternate" hreflang="{code}" href="{href}">' for code, href in counterpart.items())
        alternates += f'\n  <link rel="alternate" hreflang="x-default" href="{counterpart["en"]}">'
    lang_switch = (f'      <a class="lang-switch notranslate" translate="no" href="{switch_href}" hreflang="{other}" '
                   f'lang="{other}" title="{L["switch_title"]}">{L["switch_label"]}</a>\n') if len(LANGS) > 1 else ""
    i18n = {k: v for k, v in L["t"].items() if k.startswith("js_")}
    return Template(read("base.html")).substitute(
        lang=lang, title=html.escape(title), description=html.escape(description), body=body, prefix=prefix, assets=assets,
        favicon=FAVICON, extra_head=extra_head + alternates, repo=REPO, field_notes=FIELD_NOTES, year=date.today().year,
        canonical=counterpart[lang], course_html=L["course_html"], nav_projects=L["t"]["nav_projects"],
        toggle_theme=L["t"]["toggle_theme"], footer_data=L["t"]["footer_data"],
        lang_switch=lang_switch,
        i18n_json=json.dumps(i18n, ensure_ascii=False),
    )


def slugify(text):
    text = text.lower()
    for a, b in zip("áéíóúüñ", "aeiouun"):
        text = text.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def render_notebook(lang, m):
    nb = nbformat.read(os.path.join(ROOT, LANGS[lang]["nb_dir"], m["file"]), as_version=4)
    body, _ = HTMLExporter(template_name="basic", exclude_input_prompt=True, exclude_output_prompt=True).from_notebook_node(nb)
    # drop the notebook's own H1 (the page header shows it) and nbconvert's pilcrow anchors
    body = re.sub(r"<h1 id=[^>]*>.*?</h1>", "", body, count=1, flags=re.S)
    body = re.sub(r'<a class="anchor-link" href="#[^"]*">¶</a>', "", body)
    # code, outputs and inline code must never be machine-translated by the page translator
    body = body.replace('<div class="input_area">', '<div class="input_area notranslate" translate="no">')
    body = body.replace('<div class="output_wrapper">', '<div class="output_wrapper notranslate" translate="no">')
    body = re.sub(r"<code>", '<code class="notranslate" translate="no">', body)
    body = re.sub(r"<pre>", '<pre class="notranslate" translate="no">', body)
    # stable, readable ids for h2 headings + collect a table of contents
    toc = []

    def h2(match):
        text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        anchor = slugify(text)
        toc.append((anchor, html.unescape(text)))
        return f'<h2 id="{anchor}">{match.group(2)}</h2>'

    body = re.sub(r"<h2 id=\"([^\"]*)\">(.*?)</h2>", h2, body, flags=re.S)
    body = body.replace("<img ", f'<img loading="lazy" alt="{LANGS[lang]["t"]["fig_alt"]}" ')
    return body, toc


def module_card(lang, m):
    t = LANGS[lang]["t"]
    chips = "".join(f"<li>{html.escape(x)}</li>" for x in m["topics"])
    return f"""
      <a class="module-card" href="{m['slug']}/" data-module="{m['n']}">
        <div class="module-top"><span class="module-num">{m['n']}</span><span class="module-done" aria-hidden="true">{t['completed']}</span></div>
        <h3>{html.escape(m['title'])}</h3>
        <p>{html.escape(m['blurb'])}</p>
        <ul class="chips">{chips}</ul>
        <div class="module-meta"><span>⏱ {t['minutes'].format(m=m['minutes'])}</span><span>💻 {t['runs'].format(cpu=m['cpu'])}</span></div>
      </a>"""


def build_index(lang):
    L, mods = LANGS[lang], modules(lang)
    tracks_html = ""
    for i, (name, desc) in enumerate(L["tracks"]):
        cards = "".join(module_card(lang, m) for m in mods if m["track"] == i)
        tracks_html += f"""
    <section class="track">
      <div class="track-head"><span class="track-label">{L['t']['part'].format(n=i + 1)}</span><h2>{name}</h2><p>{desc}</p></div>
      <div class="module-grid">{cards}</div>
    </section>"""
    body = Template(read("index.html", lang)).substitute(
        tracks=tracks_html, n_modules=len(mods), hours=round(sum(m["minutes"] for m in mods) / 60), repo=REPO,
        repo_path=REPO_PATH + ("/es" if lang == "es" else ""), first=mods[0]["slug"],
        modules_json=json.dumps([m["n"] for m in mods]),
    )
    assets = "../" if L["dir"] else ""
    return page(lang, L["course"], L["description"], body, "./", assets, {k: url_of(k) for k in LANGS},
                "es/" if lang == "en" else "../")


def build_lesson(lang, i, m):
    L, t, mods = LANGS[lang], LANGS[lang]["t"], modules(lang)
    content, toc = render_notebook(lang, m)
    prev_m = mods[i - 1] if i > 0 else None
    next_m = mods[i + 1] if i + 1 < len(mods) else None
    side = "".join(
        f'<li><a href="../{x["slug"]}/" data-module="{x["n"]}" class="{"current" if x["n"] == m["n"] else ""}">'
        f'<span class="side-num">{x["n"]}</span><span class="side-title">{html.escape(x["title"])}</span></a></li>'
        for x in mods)
    toc_html = "".join(f'<li><a href="#{a}">{html.escape(x)}</a></li>' for a, x in toc)
    pager = '<nav class="pager">'
    pager += (f'<a class="prev" href="../{prev_m["slug"]}/"><small>{t["prev"]}</small><span>{prev_m["n"]} · {html.escape(prev_m["title"])}</span></a>'
              if prev_m else f'<a class="prev" href="../"><small>{t["back"]}</small><span>{t["overview_short"]}</span></a>')
    pager += (f'<a class="next" href="../{next_m["slug"]}/"><small>{t["next"]}</small><span>{next_m["n"]} · {html.escape(next_m["title"])}</span></a>'
              if next_m else f'<a class="next" href="../"><small>{t["finished"]}</small><span>{t["overview_short"]}</span></a>')
    pager += "</nav>"
    nb_path = f"{REPO_PATH}/{L['nb_dir'] + '/' if L['nb_dir'] else ''}{m['file']}"
    chips = "".join(f"<li>{html.escape(x)}</li>" for x in m["topics"])
    body = Template(read("lesson.html")).substitute(
        n=m["n"], title=html.escape(m["title"]), blurb=html.escape(m["blurb"]), chips=chips, side=side, toc=toc_html,
        content=content, pager=pager, file=m["file"],
        colab=f"https://colab.research.google.com/github/fborbon/academy/blob/main/{nb_path}",
        github=f"{REPO}/blob/main/{nb_path}", track=L["tracks"][m["track"]][0],
        module_of=t["module_of"].format(n=m["n"], total=len(mods)), meta_minutes=t["minutes"].format(m=m["minutes"]),
        meta_runs=t["runs"].format(cpu=m["cpu"]), t_download=t["download"], t_colab=t["colab"], t_github=t["github"],
        t_overview=t["overview"], t_modules=t["modules_btn"], t_on_page=t["on_page"],
        t_mark=t["js_mark"].format(n=m["n"]),
    )
    en_slug = modules("en")[i]["slug"]
    es_slug = modules("es")[i]["slug"] if "es" in LANGS else en_slug
    assets = "../../" if L["dir"] else "../"
    return page(lang, f"{m['n']} · {m['title']} | {L['course']}", m["blurb"], body, "../", assets,
                {k: url_of(k, en_slug if k == "en" else es_slug) for k in LANGS},
                f"../es/{es_slug}/" if lang == "en" else f"../../{en_slug}/", extra_head=MATHJAX)


def build():
    if os.path.exists(DIST):
        shutil.rmtree(DIST)
    os.makedirs(os.path.join(DIST, "assets"))
    shutil.copy(os.path.join(HERE, "templates", "site.css"), os.path.join(DIST, "assets", "site.css"))
    shutil.copy(os.path.join(HERE, "templates", "site.js"), os.path.join(DIST, "assets", "site.js"))
    light = HtmlFormatter(style="friendly").get_style_defs(":root .highlight")
    dark = HtmlFormatter(style="github-dark").get_style_defs(":root[data-theme=dark] .highlight")
    dark_auto = HtmlFormatter(style="github-dark").get_style_defs(":root:not([data-theme=light]) .highlight")
    with open(os.path.join(DIST, "assets", "pygments.css"), "w") as f:
        f.write(light + "\n" + dark + "\n@media (prefers-color-scheme: dark) {\n" + dark_auto + "\n}\n")
    urls = []
    for lang, L in LANGS.items():
        out = os.path.join(DIST, L["dir"])
        os.makedirs(os.path.join(out, "notebooks"), exist_ok=True)
        with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
            f.write(build_index(lang))
        urls.append(url_of(lang))
        for i, m in enumerate(modules(lang)):
            os.makedirs(os.path.join(out, m["slug"]))
            with open(os.path.join(out, m["slug"], "index.html"), "w", encoding="utf-8") as f:
                f.write(build_lesson(lang, i, m))
            shutil.copy(os.path.join(ROOT, L["nb_dir"], m["file"]), os.path.join(out, "notebooks", m["file"]))
            urls.append(url_of(lang, m["slug"]))
        print(f"  {lang}: {len(modules(lang))} lessons")
    with open(os.path.join(DIST, "robots.txt"), "w") as f:
        f.write(f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}/sitemap.xml\n")
    with open(os.path.join(DIST, "sitemap.xml"), "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                + "".join(f"  <url><loc>{u}</loc></url>\n" for u in urls) + "</urlset>\n")
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(DIST) for f in fs)
    print(f"built {len(urls)} pages into {DIST} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
