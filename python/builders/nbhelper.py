"""Helpers used by the build_NN.py scripts to generate the course notebooks.

English is the source language. Set NB_LANG=es to build the Spanish edition instead: every
markdown cell is replaced by its translation from i18n/es/NN.txt (looked up by a fingerprint of
the English text) and every code cell gets the exact text replacements listed in the same file
(comments and display labels only, never logic or data). A missing or stale translation stops
the build, so the two editions cannot silently drift apart.

Translation file format (i18n/<lang>/NN.txt):

    @@ file 01_estructuras_de_datos.ipynb
    @@ md 3f9a0c1b2e
    Spanish markdown for the cell whose English text has that fingerprint...
    @@ code
    <<< exact English fragment (one line)
    >>> its replacement
"""
import hashlib
import inspect
import os
import re

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = os.environ.get("NB_OUT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # where notebooks are written (python/ by default)
HERE = os.path.dirname(os.path.abspath(__file__))
LANG = os.environ.get("NB_LANG", "en")
_cells = []
_tr = None          # loaded translation for the current builder, when LANG != "en"


def fingerprint(text):
    return hashlib.sha1(text.strip("\n").encode("utf-8")).hexdigest()[:10]


def parse_translation(path):
    tr = {"file": None, "md": {}, "code": []}
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    mode, key, buf = None, None, []

    def flush():
        if mode == "md":
            tr["md"][key] = "\n".join(buf).strip("\n")

    for line in lines:
        m = re.match(r"^@@ (file|md|code)\s*(.*)$", line)
        if m:
            flush()
            mode, key, buf = m.group(1), m.group(2).strip(), []
            if mode == "file":
                tr["file"] = key
            continue
        if mode == "md":
            buf.append(line)
        elif mode == "code":
            if line.startswith("<<< "):
                tr["code"].append([line[4:], None, 0])
            elif line.startswith(">>> "):
                tr["code"][-1][1] = line[4:]
            elif line.strip() == ">>>":
                tr["code"][-1][1] = ""
    flush()
    return tr


def _translation():
    global _tr
    if _tr is None:
        caller = next(f.filename for f in inspect.stack() if os.path.basename(f.filename).startswith("build_"))
        num = re.search(r"build_(\d+)", os.path.basename(caller)).group(1)
        _tr = parse_translation(os.path.join(HERE, "i18n", LANG, f"{num}.txt"))
        _tr["missing"] = []
    return _tr


def md(text):
    text = text.strip("\n")
    if LANG != "en":
        tr = _translation()
        key = fingerprint(text)
        if key not in tr["md"]:
            tr["missing"].append((key, text))
            text = "**[MISSING TRANSLATION]**\n\n" + text
        else:
            text = tr["md"][key]
    _cells.append(new_markdown_cell(text))


def code(text):
    text = text.strip("\n")
    if LANG != "en":
        for rep in _translation()["code"]:
            if rep[0] in text:
                rep[2] += text.count(rep[0])
                text = text.replace(rep[0], rep[1])
    _cells.append(new_code_cell(text))


def save(name):
    if LANG != "en":
        tr = _translation()
        problems = [f"missing md translation {k}: {t[:70]!r}" for k, t in tr["missing"]]
        problems += [f"code replacement never used: {r[0]!r}" for r in tr["code"] if r[2] == 0]
        problems += [f"code replacement has no '>>>' line: {r[0]!r}" for r in tr["code"] if r[1] is None]
        if problems:
            raise SystemExit(f"{name} [{LANG}]: translation out of date\n  " + "\n  ".join(problems))
        os.makedirs(os.path.join(ROOT, LANG), exist_ok=True)
        path = os.path.join(ROOT, LANG, tr["file"])
    else:
        path = os.path.join(ROOT, name)
    nb = new_notebook(cells=list(_cells))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.write(nb, path)
    _cells.clear()
    print("wrote", path)
