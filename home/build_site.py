"""Build the Academy home page (the course catalogue) at academy.forwardforecasting.eu.

    python3 home/build_site.py            # writes home/dist/

English at the site root, Spanish under /es/. Course names, descriptions, parts, notebook counts and
study hours are read from each course's own site/build_site.py, so the catalogue never drifts from
the course websites; only the catalogue's own text lives here.
"""
import html
import importlib.util
import json
import os
import shutil
from datetime import date
from string import Template

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIST = os.path.join(HERE, "dist")
REPO = "https://github.com/fborbon/academy"
SITE_URL = "https://academy.forwardforecasting.eu"
FIELD_NOTES = "https://education.forwardforecasting.eu/"
CONTACT = "forwardforecasting@gmail.com"


def load_course(folder):
    """Import a course's site/build_site.py (it only defines data and functions at import time)."""
    path = os.path.join(ROOT, folder, "site", "build_site.py")
    spec = importlib.util.spec_from_file_location(f"course_{folder.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# catalogue order is the suggested learning order
COURSES = [
    dict(folder="python", mark="Py", en=dict(level="Beginner to advanced",
                                             needs="Having written a few lines of code. Each module builds on the previous one."),
                                     es=dict(level="De inicial a avanzado",
                                             needs="Haber escrito algunas líneas de código. Cada módulo se apoya en el anterior.")),
    dict(folder="neural-networks", mark="NN", en=dict(level="Intermediate",
                                                      needs="Python functions, loops and lists (the Intro module of Python 1001 covers them) and school algebra."),
                                              es=dict(level="Intermedio",
                                                      needs="Funciones, bucles y listas de Python (los cubre el módulo inicial de Python 1001) y álgebra de instituto.")),
]

LANGS = {
    "en": dict(
        dir="", title="Forward Forecasting Academy",
        description="Free, hands-on courses in English and Spanish: Python and neural networks as runnable notebooks, with real results, on an ordinary laptop.",
        switch_label="Español", switch_title="Ver la Academia en español",
        t=dict(
            nav_projects="Projects", toggle_theme="Toggle dark mode",
            eyebrow="Free courses · English and Spanish · no sign-up",
            lead="Every course is a set of Jupyter notebooks with their real, executed results. Read each lesson in the browser, or download it and run it on an ordinary laptop: no GPU, no API keys, no account.",
            cta_courses="Browse the courses ↓", cta_repo="Source on GitHub",
            stat_courses="courses", stat_notebooks="notebooks", stat_hours="of study", stat_langs="languages",
            catalog_h="Courses", catalog_p="Each course has its own website with a syllabus, the lessons and your progress, which is remembered in your browser. Start with Python 1001 if you are new to programming.",
            c_notebooks="{n} notebooks", c_hours="~{h} h of study", parts="Parts", needs="You need", open="Open the course →",
            path_h="Suggested path",
            path=[("Python 1001", "Learn the language, its data tools and professional practice, then use it on science, AI and forecasting problems."),
                  ("Neural Networks 1001", "Build neural networks from a single neuron to a GPT, then solve a real industrial inspection case end to end.")],
            how_h="How the courses work",
            how=[("Read it or run it", "Each lesson page shows the notebook with its saved outputs. Download it or open it in Google Colab to change the code and see what happens."),
                 ("Sized for a laptop", "Every model and dataset is chosen to run on a CPU in seconds to minutes, while still showing how the technique behaves for real."),
                 ("Honest results", "Every method is compared with a simple baseline on held-out data, and the notebooks say plainly when a technique does not help."),
                 ("Two native editions", "The English and Spanish editions are written for each language, with identical code and results, and each page links to its translation.")],
            footer=f'Questions or suggestions: <a href="mailto:{CONTACT}">{CONTACT}</a>',
        ),
    ),
    "es": dict(
        dir="es", title="Academia Forward Forecasting",
        description="Cursos gratuitos y prácticos en español e inglés: Python y redes neuronales en notebooks ejecutables, con resultados reales, en un portátil normal.",
        switch_label="English", switch_title="View the Academy in English",
        t=dict(
            nav_projects="Proyectos", toggle_theme="Cambiar a modo oscuro",
            eyebrow="Cursos gratuitos · español e inglés · sin registro",
            lead="Cada curso es un conjunto de notebooks de Jupyter con sus resultados reales ya ejecutados. Lee cada lección en el navegador, o descárgala y ejecútala en un portátil normal: sin GPU, sin claves de API y sin cuenta.",
            cta_courses="Ver los cursos ↓", cta_repo="Código en GitHub",
            stat_courses="cursos", stat_notebooks="notebooks", stat_hours="de estudio", stat_langs="idiomas",
            catalog_h="Cursos", catalog_p="Cada curso tiene su propia web con el temario, las lecciones y tu progreso, que se guarda en tu navegador. Si es tu primer contacto con la programación, empieza por Python 1001.",
            c_notebooks="{n} notebooks", c_hours="~{h} h de estudio", parts="Partes", needs="Necesitas", open="Abrir el curso →",
            path_h="Itinerario recomendado",
            path=[("Python 1001", "Aprende el lenguaje, sus herramientas de datos y la práctica profesional, y después aplícalo a problemas de ciencia, IA y previsión."),
                  ("Redes Neuronales 1001", "Construye redes neuronales desde una sola neurona hasta un GPT y resuelve de principio a fin un caso real de inspección industrial.")],
            how_h="Cómo funcionan los cursos",
            how=[("Léelo o ejecútalo", "Cada lección muestra el notebook con sus resultados guardados. Descárgalo o ábrelo en Google Colab para cambiar el código y ver qué pasa."),
                 ("A la medida de un portátil", "Cada modelo y cada conjunto de datos se eligen para ejecutarse en una CPU en segundos o minutos, sin dejar de mostrar cómo se comporta la técnica de verdad."),
                 ("Resultados honestos", "Cada método se compara con una referencia sencilla sobre datos reservados, y los notebooks dicen claramente cuándo una técnica no ayuda."),
                 ("Dos ediciones nativas", "Las ediciones en español y en inglés están escritas para cada idioma, con el mismo código y los mismos resultados, y cada página enlaza con su traducción.")],
            footer=f'Preguntas o sugerencias: <a href="mailto:{CONTACT}">{CONTACT}</a>',
        ),
    ),
}

FAVICON = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect width='64' height='64' rx='14' fill='%234f46e5'/>"
    "<path d='M14 26 32 17l18 9-18 9z' fill='%23fff'/><path d='M21 30v9c0 3 5 6 11 6s11-3 11-6v-9l-11 5z' fill='%23fff' opacity='.8'/></svg>"
)


def read(name):
    with open(os.path.join(HERE, "templates", name), encoding="utf-8") as f:
        return f.read()


def url_of(lang):
    d = LANGS[lang]["dir"]
    return SITE_URL + "/" + (d + "/" if d else "")


def course_info(c, lang):
    mod = load_course(c["folder"])
    L = mod.LANGS[lang]
    return dict(
        name=L["course"], description=L["description"], parts=[name for name, _ in L["tracks"]],
        n=len(mod.BASE), hours=round(sum(m["minutes"] for m in mod.BASE) / 60),
        url=mod.SITE_URL + "/" + (L["dir"] + "/" if L["dir"] else ""), mark=c["mark"], **c[lang],
    )


def render(lang, courses):
    L, t, other = LANGS[lang], LANGS[lang]["t"], ("es" if lang == "en" else "en")
    esc = html.escape
    cards = []
    for c in courses:
        chips = "".join(f"<li>{esc(p)}</li>" for p in c["parts"])
        cards.append(f"""      <article class="course-card">
        <div class="course-top"><span class="course-mark notranslate" translate="no">{c['mark']}</span><span class="course-level">{esc(c['level'])}</span></div>
        <h3 class="notranslate" translate="no">{esc(c['name'])}</h3>
        <p>{esc(c['description'])}</p>
        <ul class="course-meta"><li>{t["c_notebooks"].format(n=c['n'])}</li><li>{t["c_hours"].format(h=c['hours'])}</li></ul>
        <p class="course-label">{t['parts']}</p>
        <ul class="chips">{chips}</ul>
        <p class="course-label">{t['needs']}</p>
        <p class="course-needs">{esc(c['needs'])}</p>
        <a class="btn btn-primary" href="{c['url']}">{t['open']}</a>
      </article>""")
    path = "".join(f'<li><span class="step">{i}</span><div><h3 class="notranslate" translate="no">{esc(name)}</h3><p>{esc(text)}</p></div></li>'
                   for i, (name, text) in enumerate(t["path"], 1))
    how = "".join(f"<div><h3>{esc(h)}</h3><p>{esc(p)}</p></div>" for h, p in t["how"])
    counterpart = {code: url_of(code) for code in LANGS}
    alternates = "".join(f'\n  <link rel="alternate" hreflang="{code}" href="{href}">' for code, href in counterpart.items())
    alternates += f'\n  <link rel="alternate" hreflang="x-default" href="{counterpart["en"]}">'
    return Template(read("page.html")).substitute(
        lang=lang, title=esc(L["title"]), description=esc(L["description"]), canonical=url_of(lang), alternates=alternates,
        favicon=FAVICON, assets="../" if L["dir"] else "", home="./", repo=REPO, field_notes=FIELD_NOTES, year=date.today().year,
        switch_href=("../" if L["dir"] else "es/"), other_lang=other, switch_label=L["switch_label"], switch_title=L["switch_title"],
        n_courses=len(courses), n_notebooks=sum(c["n"] for c in courses), hours=sum(c["hours"] for c in courses),
        cards="\n".join(cards), path=path, how=how, **{k: (esc(v) if k != "footer" else v) for k, v in t.items() if isinstance(v, str)},
    )


def main():
    if os.path.exists(DIST):
        shutil.rmtree(DIST)
    os.makedirs(os.path.join(DIST, "assets"))
    for name in ("site.css", "site.js"):
        shutil.copy(os.path.join(HERE, "templates", name), os.path.join(DIST, "assets", name))
    for lang, L in LANGS.items():
        out = os.path.join(DIST, L["dir"])
        os.makedirs(out, exist_ok=True)
        courses = [course_info(c, lang) for c in COURSES]
        with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
            f.write(render(lang, courses))
    with open(os.path.join(DIST, "robots.txt"), "w") as f:
        f.write(f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}/sitemap.xml\n")
    urls = "".join(f"  <url><loc>{url_of(lang)}</loc></url>\n" for lang in LANGS)
    with open(os.path.join(DIST, "sitemap.xml"), "w") as f:
        f.write(f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{urls}</urlset>\n')
    print(f"Built {DIST}: " + json.dumps({lang: url_of(lang) for lang in LANGS}))


if __name__ == "__main__":
    main()
