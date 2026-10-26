"""Build the static course website for Neural Networks 1001 from the executed notebooks.

    python3 site/build_site.py            # writes site/dist/

Two editions are built from the same templates: English at the site root, Spanish under /es/
(notebooks from neural-networks/es/). The notebooks are rendered with nbconvert's "basic" template
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
REPO_PATH = "neural-networks"
SITE_URL = "https://neural-networks.academy.forwardforecasting.eu"
FIELD_NOTES = "https://fieldnotes.forwardforecasting.eu/neural-networks-1001/"

# per-module data that does not depend on the language
BASE = [
    dict(n="01", track=0, minutes=45, cpu="10 s"),
    dict(n="02", track=0, minutes=90, cpu="1 min"),
    dict(n="03", track=0, minutes=90, cpu="7 min"),
    dict(n="04", track=1, minutes=75, cpu="3 min"),
    dict(n="05", track=1, minutes=60, cpu="1 min"),
    dict(n="06", track=1, minutes=90, cpu="4 min"),
    dict(n="07", track=2, minutes=90, cpu="3 min"),
    dict(n="08", track=2, minutes=120, cpu="16 min"),
    dict(n="09", track=2, minutes=75, cpu="6 min"),
    dict(n="10", track=3, minutes=150, cpu="35 min"),
]
# modules published after the course launched: returning learners see them flagged as new
ADDED_LATER = ["10"]

LANGS = {
    "en": dict(
        dir="", nb_dir="", course="Neural Networks 1001", course_html="Neural Networks <b>1001</b>",
        description="A free, hands-on course: from a single neuron to a GPT, in ten runnable notebooks that train on a laptop CPU, ending with a capstone on a real industrial inspection case.",
        switch_label="Español", switch_title="Leer el curso en español",
        tracks=[("Foundations", "How a network computes, learns and generalises."),
                ("Classic architectures", "The inductive biases for images, time series and language."),
                ("Modern deep learning", "Attention, GPT-style language models and modern vision tasks."),
                ("Capstone", "Everything together on a real case, from the raw photos to a model you could put on a production line.")],
        modules=[
            ("01_the_neuron.ipynb", "01-the-neuron", "The Artificial Neuron",
             "Build a neuron in ten lines of NumPy, meet every activation function used in modern AI, and train it with gradient descent.",
             ["weights & bias", "activations", "vanishing gradients", "perceptron", "gradient descent"]),
            ("02_neural_network_backprop.ipynb", "02-backpropagation", "Neural Networks and Backpropagation",
             "Stack neurons into layers, write a tiny autograd engine, derive backprop by hand, check it numerically and compare optimisers.",
             ["MLP", "chain rule", "autograd", "gradient checking", "SGD, momentum, Adam", "initialisation"]),
            ("03_training_deep_networks.ipynb", "03-training-deep-networks", "Training Deep Networks Well",
             "Generalisation is the real goal: diagnose bias and variance, regularise, normalise, schedule, search hyperparameters and decide what to fix next.",
             ["train/dev/test", "bias & variance", "weight decay", "dropout", "batch norm", "LR schedules", "error analysis"]),
            ("04_convolutional_networks.ipynb", "04-convolutional-networks", "Convolutional Neural Networks",
             "Convolution from scratch, a CNN on handwritten digits, data augmentation, residual connections, transfer learning and efficient convolutions.",
             ["convolution", "pooling", "augmentation", "ResNet", "transfer learning", "depthwise-separable"]),
            ("05_recurrent_networks.ipynb", "05-recurrent-networks", "Recurrent Networks and Forecasting",
             "RNNs and LSTMs forecast hourly demand against a seasonal baseline; measure vanishing gradients through time; forecast a full day ahead.",
             ["RNN", "LSTM & GRU", "BPTT", "calendar features", "recursive vs direct forecasting"]),
            ("06_embeddings_and_seq2seq.ipynb", "06-embeddings-and-seq2seq", "Word Embeddings and Translation",
             "Train word2vec-style embeddings and find analogies, measure and remove a planted bias, then translate English to Spanish with attention and beam search.",
             ["skip-gram", "analogies", "embedding bias", "encoder-decoder", "attention", "beam search"]),
            ("07_transformer.ipynb", "07-transformer", "Building a Transformer",
             "Positional encodings, scaled dot-product and multi-head attention, masks, residual stream and LayerNorm, then a full transformer that learns to sort.",
             ["self-attention", "multi-head", "masking", "LayerNorm", "attention rollout"]),
            ("08_gpt_from_scratch.ipynb", "08-gpt-from-scratch", "A GPT from Scratch",
             "Pre-train a character-level GPT on a multi-task corpus, watch skills emerge, sample, add a KV cache, fine-tune, and see why scale matters.",
             ["tokenisation", "next-token prediction", "emergent skills", "sampling", "KV cache", "fine-tuning"]),
            ("09_vision_beyond_classification.ipynb", "09-vision-beyond-classification", "Vision Beyond Classification",
             "Segment every pixel with a U-Net, detect objects YOLO-style with non-maximum suppression, and recognise unseen classes with a triplet-loss embedding.",
             ["U-Net", "IoU", "YOLO", "NMS", "precision & recall", "triplet loss"]),
            ("10_capstone_real_case.ipynb", "10-capstone-real-case", "Capstone on a Real Case: Visual Defect Inspection",
             "Real photos of manufactured blades at the end of the line: build an inspection model the way a competition team would, from baseline to transfer learning, cross-validation, ensembling and a reject threshold.",
             ["real images", "class imbalance", "transfer learning", "cross-validation", "ensembling & TTA", "operating threshold", "Grad-CAM"]),
        ],
        t=dict(
            nav_projects="Projects", toggle_theme="Toggle dark mode",
            footer_data="Datasets are generated in the notebooks or ship with scikit-learn, except the capstone's real inspection photos (Zhou, GKN Blade Surface Defect Dataset, Mendeley Data, CC BY 4.0). Code and text: MIT licence.",
            completed="✓ completed", started="● in progress", new_badge="New", minutes="~{m} min", runs="runs in {cpu} on a CPU", part="Part {n}",
            module_of="Module {n} of {total}", download="⬇ Download notebook", colab="Open in Colab", github="View on GitHub",
            overview="← Course overview", modules_btn="Modules ▾", on_page="On this page", prev="← Previous", next="Next →",
            back="← Back to", finished="Finished? →", overview_short="Course overview", fig_alt="notebook output figure",
            js_mark="Mark module {n} as complete", js_done="✓ Module {n} completed (click to undo)",
            js_progress="{done} of {total} modules completed", js_none="Not started yet",
            js_resume="Continue with module {n}: {title} →", js_resume_new="New since you started: module {n}, {title} →",
        ),
    ),
    "es": dict(
        dir="es", nb_dir="es", course="Redes Neuronales 1001", course_html="Redes Neuronales <b>1001</b>",
        description="Un curso gratuito y práctico: de una sola neurona a un GPT, en diez notebooks ejecutables que entrenan en la CPU de un equipo personal, con un proyecto final sobre un caso real de inspección industrial.",
        switch_label="English", switch_title="Read the course in English",
        tracks=[("Fundamentos", "Cómo calcula, aprende y generaliza una red."),
                ("Arquitecturas clásicas", "Los sesgos inductivos para imágenes, series temporales y lenguaje."),
                ("Aprendizaje profundo moderno", "Atención, modelos de lenguaje tipo GPT y tareas modernas de visión."),
                ("Proyecto final", "Todo junto en un caso real, desde las fotos sin procesar hasta un modelo que podría ir a una línea de producción.")],
        modules=[
            ("01_la_neurona_artificial.ipynb", "01-la-neurona-artificial", "La neurona artificial",
             "Construye una neurona en diez líneas de NumPy, conoce todas las funciones de activación de la IA moderna y entrénala con descenso de gradiente.",
             ["pesos y sesgo", "activaciones", "gradientes que se desvanecen", "perceptrón", "descenso de gradiente"]),
            ("02_redes_neuronales_y_retropropagacion.ipynb", "02-retropropagacion", "Redes neuronales y retropropagación",
             "Apila neuronas en capas, escribe un pequeño motor de autograd, deduce la retropropagación a mano, compruébala numéricamente y compara optimizadores.",
             ["MLP", "regla de la cadena", "autograd", "comprobación de gradientes", "SGD, momentum, Adam", "inicialización"]),
            ("03_entrenar_bien_redes_profundas.ipynb", "03-entrenar-bien-redes-profundas", "Entrenar bien redes profundas",
             "La generalización es el verdadero objetivo: diagnostica sesgo y varianza, regulariza, normaliza, programa la tasa de aprendizaje, busca hiperparámetros y decide qué corregir.",
             ["entrenamiento/validación/prueba", "sesgo y varianza", "decaimiento de pesos", "dropout", "batch norm", "calendarios de LR", "análisis de errores"]),
            ("04_redes_convolucionales.ipynb", "04-redes-convolucionales", "Redes neuronales convolucionales",
             "La convolución desde cero, una CNN con dígitos manuscritos, aumento de datos, conexiones residuales, aprendizaje por transferencia y convoluciones eficientes.",
             ["convolución", "pooling", "aumento de datos", "ResNet", "transferencia", "separables en profundidad"]),
            ("05_redes_recurrentes.ipynb", "05-redes-recurrentes", "Redes recurrentes y pronóstico",
             "RNN y LSTM pronostican demanda horaria frente a una referencia estacional; mide cómo se desvanecen los gradientes en el tiempo; pronostica un día completo.",
             ["RNN", "LSTM y GRU", "BPTT", "variables de calendario", "pronóstico recursivo frente a directo"]),
            ("06_embeddings_y_traduccion.ipynb", "06-embeddings-y-traduccion", "Embeddings de palabras y traducción",
             "Entrena embeddings al estilo word2vec y encuentra analogías, mide y elimina un sesgo sembrado, y traduce de inglés a español con atención y búsqueda en haz.",
             ["skip-gram", "analogías", "sesgo en embeddings", "codificador-decodificador", "atención", "búsqueda en haz"]),
            ("07_transformer.ipynb", "07-transformer", "Construir un transformer",
             "Codificaciones posicionales, atención de producto escalar y multicabeza, máscaras, flujo residual y LayerNorm, y después un transformer completo que aprende a ordenar.",
             ["autoatención", "multicabeza", "máscaras", "LayerNorm", "rollout de atención"]),
            ("08_gpt_desde_cero.ipynb", "08-gpt-desde-cero", "Un GPT desde cero",
             "Preentrena un GPT de caracteres con un corpus multitarea, observa cómo surgen las habilidades, muestrea, añade una caché KV, haz ajuste fino y entiende por qué importa la escala.",
             ["tokenización", "predecir el siguiente token", "habilidades emergentes", "muestreo", "caché KV", "ajuste fino"]),
            ("09_vision_mas_alla_de_la_clasificacion.ipynb", "09-vision-mas-alla-de-la-clasificacion", "Visión más allá de la clasificación",
             "Segmenta cada píxel con una U-Net, detecta objetos al estilo YOLO con supresión de no máximos y reconoce clases no vistas con un embedding de pérdida triplete.",
             ["U-Net", "IoU", "YOLO", "NMS", "precisión y exhaustividad", "pérdida triplete"]),
            ("10_proyecto_final_caso_real.ipynb", "10-proyecto-final-caso-real", "Proyecto final con un caso real: inspección visual de defectos",
             "Fotos reales de álabes fabricados al final de la línea: construye un modelo de inspección como lo haría un equipo de competición, desde la referencia hasta el aprendizaje por transferencia, la validación cruzada, el ensamblado y un umbral de rechazo.",
             ["imágenes reales", "clases desbalanceadas", "transferencia", "validación cruzada", "ensamblado y TTA", "umbral de operación", "Grad-CAM"]),
        ],
        t=dict(
            nav_projects="Proyectos", toggle_theme="Cambiar a modo oscuro",
            footer_data="Los datos se generan en los notebooks o vienen incluidos en scikit-learn, salvo las fotos reales de inspección del proyecto final (Zhou, GKN Blade Surface Defect Dataset, Mendeley Data, CC BY 4.0). Código y texto: licencia MIT.",
            completed="✓ completado", started="● en curso", new_badge="Nuevo", minutes="~{m} min", runs="se ejecuta en {cpu} en una CPU", part="Parte {n}",
            module_of="Módulo {n} de {total}", download="⬇ Descargar notebook", colab="Abrir en Colab", github="Ver en GitHub",
            overview="← Presentación del curso", modules_btn="Módulos ▾", on_page="En esta página", prev="← Anterior", next="Siguiente →",
            back="← Volver a la", finished="¿Terminaste? →", overview_short="Presentación del curso", fig_alt="figura generada por el notebook",
            js_mark="Marcar el módulo {n} como completado", js_done="✓ Módulo {n} completado (clic para deshacer)",
            js_progress="{done} de {total} módulos completados", js_none="Todavía sin empezar",
            js_resume="Continúa con el módulo {n}: {title} →", js_resume_new="Nuevo desde que empezaste: módulo {n}, {title} →",
        ),
    ),
}

FAVICON = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect width='64' height='64' rx='14' fill='%234f46e5'/>"
    "<g stroke='white' stroke-width='3' opacity='.75'><line x1='16' y1='20' x2='32' y2='32'/><line x1='16' y1='44' x2='32' y2='32'/>"
    "<line x1='32' y1='32' x2='48' y2='22'/><line x1='32' y1='32' x2='48' y2='42'/></g>"
    "<g fill='white'><circle cx='16' cy='20' r='5'/><circle cx='16' cy='44' r='5'/><circle cx='32' cy='32' r='6'/>"
    "<circle cx='48' cy='22' r='5'/><circle cx='48' cy='42' r='5'/></g></svg>"
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
    L, other = LANGS[lang], ("es" if lang == "en" else "en")
    alternates = "".join(f'\n  <link rel="alternate" hreflang="{code}" href="{href}">' for code, href in counterpart.items())
    alternates += f'\n  <link rel="alternate" hreflang="x-default" href="{counterpart["en"]}">'
    i18n = {k: v for k, v in L["t"].items() if k.startswith("js_")}
    return Template(read("base.html")).substitute(
        lang=lang, title=html.escape(title), description=html.escape(description), body=body, prefix=prefix, assets=assets,
        favicon=FAVICON, extra_head=extra_head + alternates, repo=REPO, field_notes=FIELD_NOTES, year=date.today().year,
        canonical=counterpart[lang], course_html=L["course_html"], nav_projects=L["t"]["nav_projects"],
        toggle_theme=L["t"]["toggle_theme"], footer_data=L["t"]["footer_data"],
        switch_href=switch_href, switch_label=L["switch_label"], switch_title=L["switch_title"], other_lang=other,
        i18n_json=json.dumps(i18n, ensure_ascii=False), modules_json=json.dumps([b["n"] for b in BASE]),
        added_later_json=json.dumps(ADDED_LATER),
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
        <div class="module-top"><span class="module-num">{m['n']}</span><span class="module-done" aria-hidden="true">{t['completed']}</span><span class="module-started" aria-hidden="true">{t['started']}</span><span class="module-new">{t['new_badge']}</span></div>
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
    )
    assets = "../" if L["dir"] else ""
    return page(lang, L["course"], L["description"], body, "./", assets, {"en": url_of("en"), "es": url_of("es")},
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
    en_slug, es_slug = modules("en")[i]["slug"], modules("es")[i]["slug"]
    assets = "../../" if L["dir"] else "../"
    return page(lang, f"{m['n']} · {m['title']} | {L['course']}", m["blurb"], body, "../", assets,
                {"en": url_of("en", en_slug), "es": url_of("es", es_slug)},
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
