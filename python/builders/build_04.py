from nbhelper import md, code, save

md(r"""
# 04 · Environments, Packages, Containers and Editors

Code that only runs on your machine is not finished. This module covers the practices that make Python
projects reproducible and deployable: isolated environments, package managers, lock files, containers,
and the editors professionals write code in.

**You will learn**
1. Why every project needs its own environment, and how `venv` works (demonstrated live)
2. Package managers compared: `pip`, `uv`, `pipx`, `conda`/`mamba`, and the system ones (`apt`, `snap`, `brew`, `winget`)
3. Pinning dependencies: requirements files, lock files and `pyproject.toml`
4. Docker: packaging an application with its whole environment, and deploying it
5. Editors and IDEs: VS Code, PyCharm, Eclipse with PyDev, Vim/Neovim, JupyterLab, Spyder
6. A project layout and habits worth adopting from day one
""")

md(r"""
## 1. Which Python am I running?
There is rarely just one Python on a machine: the operating system's, one from a package manager, one
per project. Always know which one runs your code.
""")
code(r"""
import sys, platform, site
print("interpreter :", sys.executable)
print("version     :", platform.python_version())
print("prefix      :", sys.prefix)               # the environment this interpreter belongs to
print("base prefix :", sys.base_prefix)          # the installation it was created from
print("in a venv   :", sys.prefix != sys.base_prefix)
""")

md(r"""
## 2. Isolated environments with `venv`
Project A needs `pandas 1.5`, project B needs `pandas 2.2`: installed globally, one of them breaks. A
**virtual environment** is a folder with its own `site-packages`, so each project gets exactly the
versions it needs and nothing leaks between them. Created live below:
""")
code(r"""
import subprocess, tempfile, os, venv

env_dir = os.path.join(tempfile.mkdtemp(), "demo-env")
venv.create(env_dir, with_pip=True)                         # same as: python -m venv demo-env
py = os.path.join(env_dir, "bin", "python") if os.name != "nt" else os.path.join(env_dir, "Scripts", "python.exe")

def in_env(*args):
    return subprocess.run([py, *args], capture_output=True, text=True).stdout.strip()

print("new environment's prefix:", in_env("-c", "import sys; print(sys.prefix)"))
print("packages it contains    :", in_env("-m", "pip", "list", "--format=freeze") or "(only pip)")
print("can it import pandas?   :", in_env("-c", "import importlib.util as u; print(u.find_spec('pandas') is not None)"))
print("this kernel has pandas  :", __import__("importlib.util").util.find_spec("pandas") is not None)
""")

md(r"""
The new environment is empty: it cannot see the libraries installed for this notebook. That is the point.
Day to day, from a terminal:

```bash
python3 -m venv .venv              # create (once per project)
source .venv/bin/activate          # activate (Windows: .venv\Scripts\activate)
pip install -r requirements.txt    # install the project's dependencies
deactivate                         # leave
```

Add `.venv/` to `.gitignore`: environments are rebuilt from the dependency list, never committed.
""")

md(r"""
## 3. Package managers
| Tool | Installs | Use it for |
|---|---|---|
| **pip** | Python packages from PyPI | the standard; installs into the active environment |
| **uv** | Python packages and Python itself | a much faster drop-in for pip/venv/pip-tools, with lock files (`uv lock`, `uv sync`) |
| **pipx** | command-line tools, each in its own hidden venv | installing tools like `ruff`, `black`, `httpie` without polluting projects |
| **conda / mamba** | Python *and non-Python* packages (C/Fortran libraries, CUDA, R) from conda-forge | scientific stacks with heavy native dependencies, GPU work |
| **Poetry / Hatch / PDM** | project managers built on pyproject.toml | dependency resolution, lock files, building and publishing in one tool |
| **apt / dnf** (Linux) | system packages | the operating system's own software, compilers, system libraries |
| **snap / flatpak** (Linux) | sandboxed applications | desktop apps and tools that bundle their own dependencies |
| **brew** (macOS/Linux), **winget / choco** (Windows) | system packages | the same role as apt on other platforms |

Good habits:
- Never `sudo pip install`: it can break tools your operating system depends on. Use a venv or pipx.
- Do not mix pip and conda in the same environment unless you know why; prefer one.
- Install Python itself with your system's package manager, `pyenv`, `uv python install`, or the official installer.
""")
code(r"""
r = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"], capture_output=True, text=True)
import json
pkgs = {p["name"].lower(): p["version"] for p in json.loads(r.stdout)}
print(f"{len(pkgs)} packages in this kernel's environment, for example:")
for name in ["numpy", "pandas", "polars", "duckdb", "torch", "tensorflow"]:
    print(f"  {name:<11}{pkgs.get(name, 'not installed')}")
""")

md(r"""
## 4. Pinning dependencies
A project that worked last year may not install today: a dependency released a breaking version. Record
what you rely on.

| File | What it pins | Example |
|---|---|---|
| `requirements.txt` with ranges | compatible versions | `pandas>=2.2,<3` |
| `requirements.lock` / `pip freeze` | the **exact** versions that were tested | `pandas==2.2.3` |
| `pyproject.toml` | a package's own dependencies (module 07) | `dependencies = ["pandas>=2.2"]` |
| `uv.lock`, `poetry.lock` | exact versions plus hashes for every platform | generated by the tool |

Libraries declare ranges (so they combine with other libraries); applications deploy from exact locks
(so production runs what was tested).
""")
code(r"""
frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
print("\n".join(frozen[:8]), f"\n... {len(frozen)} lines in total")
""")

md(r"""
## 5. Containers: shipping the environment, not just the code
A virtual environment isolates Python packages; a **container** isolates everything: the operating system
libraries, the Python version, your code and its configuration. The same image runs on your laptop, a
colleague's machine and a cloud server, which is why most web services are deployed this way.

A Dockerfile for a small web API:
""")
code(r'''
dockerfile = """\
# small official base image with Python
FROM python:3.12-slim

WORKDIR /app
# dependencies first: this layer is cached between builds while only the code changes
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# then the code, which changes often
COPY . .
# do not run as root
USER 1000
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
compose = """\
services:
  api:
    build: .
    ports: ["127.0.0.1:8000:8000"]    # only reachable from this machine (a web server proxies to it)
    env_file: .env                     # secrets stay out of the image
    restart: unless-stopped
    mem_limit: 512m                    # a runaway process cannot take the whole server
"""
print(dockerfile); print(compose)
''')
code(r"""
import shutil
docker = shutil.which("docker")
if docker:
    print(subprocess.run([docker, "--version"], capture_output=True, text=True).stdout.strip())
else:
    print("docker is not installed here (normal on Colab). The commands to try on your own machine:")
print('''
docker build -t my-api .          # build the image from the Dockerfile
docker run -p 8000:8000 my-api    # run one container
docker compose up -d              # run the services described in compose.yaml, in the background
docker compose logs -f api        # follow the logs
''')
""")

md(r"""
**Container good practices:** start from a `slim` image; copy and install dependencies before the code so
rebuilds are fast; run as a non-root user; keep secrets in environment variables or files mounted at run
time, never baked into the image; set memory limits; one process per container; store data in volumes,
not inside the container.

| | virtual environment | container |
|---|---|---|
| isolates | Python packages | OS libraries, Python, packages, config |
| size | tens to hundreds of MB | hundreds of MB to a few GB |
| best for | development, scripts | deployment, services, reproducing a full stack |

## 6. Editors and IDEs
| Tool | Type | Strong points |
|---|---|---|
| **Visual Studio Code** | free editor with extensions | Python and Jupyter extensions, debugger, remote development over SSH and in containers, Git integration. The most used today. |
| **PyCharm** | full IDE (free Community edition) | deep code analysis and refactoring, test runner, database tools |
| **Eclipse + PyDev** | IDE (free) | familiar to Java teams; debugger, refactoring, Django support |
| **Vim / Neovim** | terminal editor | available on every server, extremely fast once learned; Neovim plus a language server gives IDE features |
| **JupyterLab** | notebook environment in the browser | exploration, data analysis, teaching (this course) |
| **Spyder** | scientific IDE | MATLAB-like variable explorer, popular in science and engineering |
| **Visual Studio** (Windows) | full IDE | Python workload for teams already on Microsoft tooling |

Whatever you choose, learn three things in it: **run the debugger** with breakpoints, **jump to definition**,
and **rename a symbol** across the project. Knowing basic Vim is worth it too: it is often the only editor on
a remote server.

## 7. A project layout to start from
```
my-project/
├── .venv/                 # environment (not committed)
├── src/my_project/        # the code, importable as a package
│   ├── __init__.py
│   └── core.py
├── tests/                 # pytest tests (module 06)
├── notebooks/             # exploration, not production code
├── pyproject.toml         # metadata, dependencies, tool settings (module 07)
├── requirements.lock      # exact versions for deployment
├── Dockerfile             # if it is deployed as a service
├── .gitignore             # .venv/, __pycache__/, .env, data files
├── .env.example           # documents the settings; the real .env is never committed
└── README.md
```

Habits that pay off: a formatter and linter (`ruff format`, `ruff check`) run automatically with
`pre-commit`; type hints checked by `mypy` or `pyright`; tests run on every push by CI (GitHub Actions);
secrets only in environment variables.

## Exercises
1. Create a venv for a small project, install `requests`, and write a `requirements.lock` with `pip freeze`. Delete the venv and rebuild it from the lock.
2. Install `ruff` with `pipx` and run `ruff check` on one of your scripts.
3. Write a Dockerfile for a script that prints the current date; build and run it.
""")

save("04_environments_and_tools.ipynb")
