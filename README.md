# Academy

Free, hands-on courses, listed on the Academy home page at [academy.forwardforecasting.eu](https://academy.forwardforecasting.eu/) (Spanish at [/es/](https://academy.forwardforecasting.eu/es/)). Every course is a set of runnable Jupyter notebooks with their results saved, readable on GitHub and runnable on an ordinary laptop CPU, with no GPU and no API keys.

| Course | Folder | Website |
|---|---|---|
| **Neural Networks 1001**: from a single neuron to a GPT in nine notebooks, plus a capstone on a real industrial defect-inspection case | [`neural-networks/`](neural-networks/) | [neural-networks.academy.forwardforecasting.eu](https://neural-networks.academy.forwardforecasting.eu/) |
| **Redes Neuronales 1001**: the same course in Spanish | [`neural-networks/es/`](neural-networks/es/) | [neural-networks.academy.forwardforecasting.eu/es/](https://neural-networks.academy.forwardforecasting.eu/es/) |
| **Python 1001**: Intro, Medium and Advanced modules, from data structures to frontier science, forecasting and a Kaggle capstone, in thirteen notebooks | [`python/`](python/) | [python.academy.forwardforecasting.eu](https://python.academy.forwardforecasting.eu/) |
| **Python 1001 (español)**: the same course in Spanish | [`python/es/`](python/es/) | [python.academy.forwardforecasting.eu/es/](https://python.academy.forwardforecasting.eu/es/) |
| **RAG Systems 1001**: retrieval-augmented generation you can trust, in eleven notebooks: hybrid retrieval, frameworks, agents, evaluation and a supervisor model | [`rag-systems/`](rag-systems/) | [rag-systems.academy.forwardforecasting.eu](https://rag-systems.academy.forwardforecasting.eu/) |
| **RAG Systems 1001 (español)**: the same course in Spanish | [`rag-systems/es/`](rag-systems/es/) | [rag-systems.academy.forwardforecasting.eu/es/](https://rag-systems.academy.forwardforecasting.eu/es/) |

## Running a course

Each course folder has its own README with the syllabus and `requirements.txt`.

```
cd python            # or neural-networks
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
jupyter lab          # open the notebooks in order
```

The notebooks are saved with their results, so you can read them straight on GitHub before running anything. The Spanish editions are in the `es/` subfolders.

## Contact

Questions or suggestions: forwardforecasting@gmail.com
