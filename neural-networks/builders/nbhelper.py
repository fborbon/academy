import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the neural-networks/ folder
_cells = []


def md(text):
    _cells.append(new_markdown_cell(text.strip("\n")))


def code(text):
    _cells.append(new_code_cell(text.strip("\n")))


def save(name):
    path = os.path.join(ROOT, name)
    nb = new_notebook(cells=list(_cells))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.write(nb, path)
    _cells.clear()
    print("wrote", path)
