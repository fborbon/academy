"""Print the translatable parts of one notebook, to write or update its translation file.

    python3 builders/i18n_skeleton.py 04            # every markdown cell, with fingerprints
    python3 builders/i18n_skeleton.py 04 --missing  # only cells the es translation lacks
    python3 builders/i18n_skeleton.py 04 --code     # code cells, to pick comments/labels to translate
"""
import os
import sys

import nbhelper

num, flags = sys.argv[1], sys.argv[2:]
lang = os.environ.get("NB_LANG_TARGET", "es")
captured = []
nbhelper.md = lambda text: captured.append(("md", text.strip("\n")))
nbhelper.code = lambda text: captured.append(("code", text.strip("\n")))
nbhelper.save = lambda name: None
src = open(os.path.join(nbhelper.HERE, f"build_{num}.py"), encoding="utf-8").read()
exec(compile(src.replace("from nbhelper import md, code, save", "from nbhelper import md, code, save\nmd, code, save = nbhelper.md, nbhelper.code, nbhelper.save"), f"build_{num}.py", "exec"), {"nbhelper": nbhelper})

done = {}
path = os.path.join(nbhelper.HERE, "i18n", lang, f"{num}.txt")
if os.path.exists(path):
    done = nbhelper.parse_translation(path)["md"]
for kind, text in captured:
    if "--code" in flags:
        if kind == "code":
            print("@@ ----- code cell -----\n" + text)
        continue
    if kind == "md" and ("--missing" not in flags or nbhelper.fingerprint(text) not in done):
        print(f"@@ md {nbhelper.fingerprint(text)}\n{text}\n")
