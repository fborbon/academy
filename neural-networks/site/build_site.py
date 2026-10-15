"""Build the static course website for Neural Networks 1001 from the executed notebooks.

    python3 site/build_site.py            # writes site/dist/

The notebooks are rendered with nbconvert's "basic" template (body only) and wrapped in the
course layout from site/templates. Nothing is fetched at build time.
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
FIELD_NOTES = "https://education.forwardforecasting.eu/neural-networks-1001/"

MODULES = [
    dict(n="01", file="01_the_neuron.ipynb", slug="01-the-neuron", track=0, minutes=45, cpu="10 s",
         title="The Artificial Neuron",
         blurb="Build a neuron in ten lines of NumPy, meet every activation function used in modern AI, and train it with gradient descent.",
         topics=["weights & bias", "activations", "vanishing gradients", "perceptron", "gradient descent"]),
    dict(n="02", file="02_neural_network_backprop.ipynb", slug="02-backpropagation", track=0, minutes=90, cpu="1 min",
         title="Neural Networks and Backpropagation",
         blurb="Stack neurons into layers, write a tiny autograd engine, derive backprop by hand, check it numerically and compare optimisers.",
         topics=["MLP", "chain rule", "autograd", "gradient checking", "SGD, momentum, Adam", "initialisation"]),
    dict(n="03", file="03_training_deep_networks.ipynb", slug="03-training-deep-networks", track=0, minutes=90, cpu="7 min",
         title="Training Deep Networks Well",
         blurb="Generalisation is the real goal: diagnose bias and variance, regularise, normalise, schedule, search hyperparameters and decide what to fix next.",
         topics=["train/dev/test", "bias & variance", "weight decay", "dropout", "batch norm", "LR schedules", "error analysis"]),
    dict(n="04", file="04_convolutional_networks.ipynb", slug="04-convolutional-networks", track=1, minutes=75, cpu="3 min",
         title="Convolutional Neural Networks",
         blurb="Convolution from scratch, a CNN on handwritten digits, data augmentation, residual connections, transfer learning and efficient convolutions.",
         topics=["convolution", "pooling", "augmentation", "ResNet", "transfer learning", "depthwise-separable"]),
    dict(n="05", file="05_recurrent_networks.ipynb", slug="05-recurrent-networks", track=1, minutes=60, cpu="1 min",
         title="Recurrent Networks and Forecasting",
         blurb="RNNs and LSTMs forecast hourly demand against a seasonal baseline; measure vanishing gradients through time; forecast a full day ahead.",
         topics=["RNN", "LSTM & GRU", "BPTT", "calendar features", "recursive vs direct forecasting"]),
    dict(n="06", file="06_embeddings_and_seq2seq.ipynb", slug="06-embeddings-and-seq2seq", track=1, minutes=90, cpu="4 min",
         title="Word Embeddings and Translation",
         blurb="Train word2vec-style embeddings and find analogies, measure and remove a planted bias, then translate English to Spanish with attention and beam search.",
         topics=["skip-gram", "analogies", "embedding bias", "encoder-decoder", "attention", "beam search"]),
    dict(n="07", file="07_transformer.ipynb", slug="07-transformer", track=2, minutes=90, cpu="3 min",
         title="Building a Transformer",
         blurb="Positional encodings, scaled dot-product and multi-head attention, masks, residual stream and LayerNorm, then a full transformer that learns to sort.",
         topics=["self-attention", "multi-head", "masking", "LayerNorm", "attention rollout"]),
    dict(n="08", file="08_gpt_from_scratch.ipynb", slug="08-gpt-from-scratch", track=2, minutes=120, cpu="16 min",
         title="A GPT from Scratch",
         blurb="Pre-train a character-level GPT on a multi-task corpus, watch skills emerge, sample, add a KV cache, fine-tune, and see why scale matters.",
         topics=["tokenisation", "next-token prediction", "emergent skills", "sampling", "KV cache", "fine-tuning"]),
    dict(n="09", file="09_vision_beyond_classification.ipynb", slug="09-vision-beyond-classification", track=2, minutes=75, cpu="6 min",
         title="Vision Beyond Classification",
         blurb="Segment every pixel with a U-Net, detect objects YOLO-style with non-maximum suppression, and recognise unseen classes with a triplet-loss embedding.",
         topics=["U-Net", "IoU", "YOLO", "NMS", "precision & recall", "triplet loss"]),
]
TRACKS = [
    ("Foundations", "How a network computes, learns and generalises."),
    ("Classic architectures", "The inductive biases for images, time series and language."),
    ("Modern deep learning", "Attention, GPT-style language models and modern vision tasks."),
]

FAVICON = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect width='64' height='64' rx='14' fill='%234f46e5'/>"
    "<g stroke='white' stroke-width='3' opacity='.75'><line x1='16' y1='20' x2='32' y2='32'/><line x1='16' y1='44' x2='32' y2='32'/>"
    "<line x1='32' y1='32' x2='48' y2='22'/><line x1='32' y1='32' x2='48' y2='42'/></g>"
    "<g fill='white'><circle cx='16' cy='20' r='5'/><circle cx='16' cy='44' r='5'/><circle cx='32' cy='32' r='6'/>"
    "<circle cx='48' cy='22' r='5'/><circle cx='48' cy='42' r='5'/></g></svg>"
)


def read(name):
    with open(os.path.join(HERE, "templates", name), encoding="utf-8") as f:
        return f.read()


def page(title, description, body, depth, extra_head=""):
    prefix = "../" * depth
    return Template(read("base.html")).substitute(
        title=html.escape(title), description=html.escape(description), body=body, prefix=prefix,
        favicon=FAVICON, extra_head=extra_head, repo=REPO, field_notes=FIELD_NOTES, year=date.today().year,
        site_url=SITE_URL,
    )


def total_hours():
    return round(sum(m["minutes"] for m in MODULES) / 60)


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def render_notebook(m):
    nb = nbformat.read(os.path.join(ROOT, m["file"]), as_version=4)
    body, _ = HTMLExporter(template_name="basic", exclude_input_prompt=True, exclude_output_prompt=True).from_notebook_node(nb)
    # drop the notebook's own H1 (the page header shows it) and nbconvert's pilcrow anchors
    body = re.sub(r"<h1 id=[^>]*>.*?</h1>", "", body, count=1, flags=re.S)
    body = re.sub(r'<a class="anchor-link" href="#[^"]*">¶</a>', "", body)
    # stable, readable ids for h2 headings + collect a table of contents
    toc = []

    def h2(match):
        text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        anchor = slugify(text)
        toc.append((anchor, html.unescape(text)))
        return f'<h2 id="{anchor}">{match.group(2)}</h2>'

    body = re.sub(r"<h2 id=\"([^\"]*)\">(.*?)</h2>", h2, body, flags=re.S)
    body = body.replace("<img ", '<img loading="lazy" alt="notebook output figure" ')
    return body, toc


def module_card(m, prefix):
    chips = "".join(f"<li>{html.escape(t)}</li>" for t in m["topics"])
    return f"""
      <a class="module-card" href="{prefix}{m['slug']}/" data-module="{m['n']}">
        <div class="module-top"><span class="module-num">{m['n']}</span><span class="module-done" aria-hidden="true">✓ completed</span></div>
        <h3>{html.escape(m['title'])}</h3>
        <p>{html.escape(m['blurb'])}</p>
        <ul class="chips">{chips}</ul>
        <div class="module-meta"><span>⏱ ~{m['minutes']} min</span><span>💻 runs in {m['cpu']} on a CPU</span></div>
      </a>"""


def build_index():
    tracks_html = ""
    for t, (name, desc) in enumerate(TRACKS):
        cards = "".join(module_card(m, "") for m in MODULES if m["track"] == t)
        tracks_html += f"""
    <section class="track">
      <div class="track-head"><span class="track-label">Part {t + 1}</span><h2>{name}</h2><p>{desc}</p></div>
      <div class="module-grid">{cards}</div>
    </section>"""
    body = Template(read("index.html")).substitute(
        tracks=tracks_html, n_modules=len(MODULES), hours=total_hours(), repo=REPO, repo_path=REPO_PATH,
        first=MODULES[0]["slug"], modules_json=json.dumps([m["n"] for m in MODULES]),
    )
    return page("Neural Networks 1001",
                "A free, hands-on course: from a single neuron to a GPT, in nine runnable notebooks that train on a laptop CPU.",
                body, 0)


def build_lesson(i, m):
    content, toc = render_notebook(m)
    prev_m = MODULES[i - 1] if i > 0 else None
    next_m = MODULES[i + 1] if i + 1 < len(MODULES) else None
    side = "".join(
        f'<li><a href="../{x["slug"]}/" data-module="{x["n"]}" class="{"current" if x is m else ""}">'
        f'<span class="side-num">{x["n"]}</span><span class="side-title">{html.escape(x["title"])}</span></a></li>'
        for x in MODULES)
    toc_html = "".join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in toc)
    pager = '<nav class="pager">'
    pager += (f'<a class="prev" href="../{prev_m["slug"]}/"><small>← Previous</small><span>{prev_m["n"]} · {html.escape(prev_m["title"])}</span></a>'
              if prev_m else '<a class="prev" href="../"><small>← Back to</small><span>Course overview</span></a>')
    pager += (f'<a class="next" href="../{next_m["slug"]}/"><small>Next →</small><span>{next_m["n"]} · {html.escape(next_m["title"])}</span></a>'
              if next_m else '<a class="next" href="../"><small>Finished? →</small><span>Course overview</span></a>')
    pager += "</nav>"
    colab = f"https://colab.research.google.com/github/fborbon/academy/blob/main/{REPO_PATH}/{m['file']}"
    chips = "".join(f"<li>{html.escape(t)}</li>" for t in m["topics"])
    body = Template(read("lesson.html")).substitute(
        n=m["n"], title=html.escape(m["title"]), blurb=html.escape(m["blurb"]), minutes=m["minutes"], cpu=m["cpu"],
        chips=chips, side=side, toc=toc_html, content=content, pager=pager, file=m["file"], colab=colab,
        github=f"{REPO}/blob/main/{REPO_PATH}/{m['file']}", track=TRACKS[m["track"]][0], total=len(MODULES),
    )
    mathjax = """
  <script>
    window.MathJax = { tex: { inlineMath: [["$", "$"], ["\\\\(", "\\\\)"]], displayMath: [["$$", "$$"], ["\\\\[", "\\\\]"]], processEscapes: true },
                       options: { skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"] },
                       chtml: { displayAlign: "center" } };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>"""
    return page(f"{m['n']} · {m['title']} | Neural Networks 1001", m["blurb"], body, 1, extra_head=mathjax)


def build():
    if os.path.exists(DIST):
        shutil.rmtree(DIST)
    os.makedirs(os.path.join(DIST, "assets"))
    os.makedirs(os.path.join(DIST, "notebooks"))
    shutil.copy(os.path.join(HERE, "templates", "site.css"), os.path.join(DIST, "assets", "site.css"))
    shutil.copy(os.path.join(HERE, "templates", "site.js"), os.path.join(DIST, "assets", "site.js"))
    light = HtmlFormatter(style="friendly").get_style_defs(":root .highlight")
    dark = HtmlFormatter(style="github-dark").get_style_defs(":root[data-theme=dark] .highlight")
    dark_auto = HtmlFormatter(style="github-dark").get_style_defs(":root:not([data-theme=light]) .highlight")
    with open(os.path.join(DIST, "assets", "pygments.css"), "w") as f:
        f.write(light + "\n" + dark + "\n@media (prefers-color-scheme: dark) {\n" + dark_auto + "\n}\n")
    with open(os.path.join(DIST, "index.html"), "w", encoding="utf-8") as f:
        f.write(build_index())
    for i, m in enumerate(MODULES):
        os.makedirs(os.path.join(DIST, m["slug"]))
        with open(os.path.join(DIST, m["slug"], "index.html"), "w", encoding="utf-8") as f:
            f.write(build_lesson(i, m))
        shutil.copy(os.path.join(ROOT, m["file"]), os.path.join(DIST, "notebooks", m["file"]))
        print(f"  {m['slug']}")
    with open(os.path.join(DIST, "robots.txt"), "w") as f:
        f.write(f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}/sitemap.xml\n")
    urls = [SITE_URL + "/"] + [f"{SITE_URL}/{m['slug']}/" for m in MODULES]
    with open(os.path.join(DIST, "sitemap.xml"), "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                + "".join(f"  <url><loc>{u}</loc></url>\n" for u in urls) + "</urlset>\n")
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(DIST) for f in fs)
    print(f"built {len(MODULES)} lessons into {DIST} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
