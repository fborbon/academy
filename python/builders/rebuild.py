"""Regenerate notebooks from their builders without losing saved outputs.

    python3 builders/rebuild.py 04                  # English: regenerate, keep outputs if the code is unchanged
    python3 builders/rebuild.py 04 --lang es        # same for the Spanish edition
    python3 builders/rebuild.py 04 --lang es --execute   # regenerate and re-run (needed after code changes)
    python3 builders/rebuild.py all --check         # CI: committed notebooks match their builders (every edition present)

Run the builders through this script rather than directly: a builder on its own writes a fresh
notebook with no outputs. Execution runs one notebook at a time on purpose (parallel PyTorch or
TensorFlow kernels oversubscribe the CPU and each one becomes many times slower).
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# editions that exist (the Spanish edition is added by creating builders/i18n/es/)
LANGS = ["en"] + (["es"] if os.path.isdir(os.path.join(HERE, "i18n", "es")) else [])
ALL = sorted(os.path.basename(p)[6:8] for p in glob.glob(os.path.join(HERE, "build_[0-9][0-9].py")))


def build(num, lang, out_dir):
    env = dict(os.environ, NB_LANG=lang, NB_OUT_DIR=out_dir)
    r = subprocess.run([sys.executable, os.path.join(HERE, f"build_{num}.py")], env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout + r.stderr)
    path = r.stdout.strip().split("wrote ", 1)[1].strip()
    return path, os.path.relpath(path, out_dir)


def code_sources(nb):
    return [c.source for c in nb.cells if c.cell_type == "code"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nums", nargs="+", help="notebook numbers, or 'all'")
    ap.add_argument("--lang", default="en", choices=["en", "es"])
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--check", action="store_true", help="verify only (both languages); writes nothing")
    a = ap.parse_args()
    nums = ALL if a.nums == ["all"] else [n.zfill(2) for n in a.nums]
    langs = LANGS if a.check else [a.lang]
    failed = []
    for lang in langs:
        for num in nums:
            with tempfile.TemporaryDirectory() as tmp:
                fresh_path, rel = build(num, lang, tmp)
                fresh = nbformat.read(fresh_path, 4)
            target = os.path.join(ROOT, rel)
            old = nbformat.read(target, 4) if os.path.exists(target) else None
            if a.check:
                if old is None or [(c.cell_type, c.source) for c in old.cells] != [(c.cell_type, c.source) for c in fresh.cells]:
                    failed.append(f"{rel}: differs from its builder/translation (run rebuild.py {num} --lang {lang})")
                continue
            same_code = old is not None and code_sources(old) == code_sources(fresh)
            if same_code and not a.execute:
                for o, n in zip([c for c in old.cells if c.cell_type == "code"], [c for c in fresh.cells if c.cell_type == "code"]):
                    n.outputs, n.execution_count, n.metadata = o.outputs, o.execution_count, o.metadata
                if len(old.cells) == len(fresh.cells):          # keep cell ids stable to avoid noisy diffs
                    for o, n in zip(old.cells, fresh.cells):
                        n.id = o.id
                fresh.metadata = old.metadata
                nbformat.write(fresh, target)
                print(f"{rel}: regenerated, outputs kept")
                continue
            nbformat.write(fresh, target)
            if not a.execute:
                print(f"{rel}: code changed, written WITHOUT outputs; re-run with --execute")
                continue
            env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "6"))
            subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            "--ExecutePreprocessor.timeout=3600", target], check=True, env=env, cwd=os.path.dirname(target))
            print(f"{rel}: regenerated and executed")
    if failed:
        raise SystemExit("\n".join(failed))
    if a.check:
        print(f"OK: {len(nums)} notebooks x {len(langs)} languages match their builders")


if __name__ == "__main__":
    main()
