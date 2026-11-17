# RAG Systems 1001

A free, hands-on course on building **Retrieval-Augmented Generation (RAG)** systems that you can trust:
how to find the right passages, answer only from them, measure the answers, and let a second model
(a *supervisor*) judge every answer so the system improves continuously. The lessons are Jupyter notebooks
with their results saved, so they can be read on the website, and they run on an ordinary laptop CPU with no
GPU and no API keys.

**Website:** [rag-systems.academy.forwardforecasting.eu](https://rag-systems.academy.forwardforecasting.eu/) (Spanish edition under `/es/`, notebooks in [`es/`](es/))

**Status:** all eleven modules are written and executed, in English and Spanish.

## Table of contents
1. [Syllabus](#syllabus)
2. [Where the material comes from](#where-the-material-comes-from)
3. [Run it yourself](#run-it-yourself)
4. [Technologies covered](#technologies-covered)
5. [Data and licences](#data-and-licences)
6. [Cost and resources](#cost-and-resources)

## Syllabus

Every module uses one small fictional document set, the operations handbook of an offshore wind farm
(*Harbourview*), written for the course, plus a battery of questions with known answers. Because the whole
corpus can be read in ten minutes, you can always see *why* a retrieval or an answer failed.

| # | Notebook | Module | What it covers | Status |
|---|---|---|---|---|
| 01 | From a Question to a Grounded Answer | Foundations | the anatomy of RAG; chunking measured (fixed windows vs. the document's structure); BM25 written from scratch; hit@k and MRR; grounded answers with citations; refusing when the documents are silent; the two kinds of failure | done |
| 02 | Semantic and Hybrid Search | Foundations | embeddings and cosine similarity (MiniLM on CPU); lexical vs. semantic vs. hybrid on paraphrases and exact codes; Reciprocal Rank Fusion and its weights; calibrating refusal; when a vector index (FAISS) is worth it, measured | done |
| 03 | LangChain and LlamaIndex | Building | the same corpus through two frameworks (document loaders, splitters, vector stores, retrievers), what each abstracts and costs, when to skip both | done |
| 04 | Model Routing and Typed Output | Building | LiteLLM as one interface to many models (Bedrock Nova behind it), Pydantic schemas for requests, responses and structured model output, validation and retries | done |
| 05 | Agents with LangGraph | Building | a RAG pipeline as a state graph: nodes, conditional routing, parallel branches, tools, checkpoints; the forecast-diagnose-retrieve-recommend graph pattern | done |
| 06 | Answers from Data: Safe Text-to-SQL | Building | when the answer is a number, not a passage: a model writes one read-only SELECT over views, a guard validates it, DuckDB runs it, the result is cited like a passage | done |
| 07 | Evaluating a RAG System | Quality | a question battery with references taken from the data, regex grading, LLM-as-judge, faithfulness and citation checks, sizing an evaluation, before/after fixes | done |
| 08 | The Supervisor Model | Quality | a second model judges every answer (verdict, faithfulness, citations, retrieval miss, what was missing); a bank of validated answers; user remarks judged and turned into documentation fixes; the continuous-improvement loop | done |
| 09 | Observability: Langfuse and LangSmith | Production | traces, token and cost accounting, scores attached to traces, dashboards; why one tracing stack was enough | done |
| 10 | Production RAG on a Budget | Production | a serverless search endpoint (BM25 + one cheap model call), memory limits, cost per question, security and rate limits; **GraphRAG and Neo4j**: when a knowledge graph pays and why we did not use one | done |
| 11 | Capstone: A Supervised RAG Assistant | Capstone | build the whole system end to end on a new corpus, evaluate it, run the supervisor, and fix what it finds | done |

## Where the material comes from

The course teaches what was actually built and measured in this portfolio's projects, in generic form:

* **A project assistant with a supervisor model** (a chat over a forecasting project's documentation and
  data): hybrid BM25 + embeddings retrieval fused by RRF, read-only SQL over result tables, an answer bank,
  a daily judge that grades every answer, and a loop that turns user corrections into documentation fixes
  (modules 02, 05, 06, 08).
* **A wind-farm analytics agent** (live): a LangGraph pipeline, two independent retrieval stacks (LangChain
  with FAISS, LlamaIndex), LiteLLM routing to Amazon Nova, Pydantic schemas, Langfuse tracing and a graded
  35-question evaluation that took the agent from 5/15 to 13/15 correct on its first set (modules 03 to 05, 07, 09).
* **A news and social-signal site**: BM25 retrieval in an AWS Lambda that scales to zero, with a small model
  answering from the retrieved posts only (module 10).

No company, customer, colleague or confidential figure appears in the course; the examples are fictional
and every result printed in a notebook is computed in the notebook.

## Run it yourself

```bash
git clone https://github.com/fborbon/academy.git
cd academy/rag-systems
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
jupyter lab
```

Module 02 downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB) from Hugging Face on first run.
Later modules add their packages to `requirements.txt` as they land.

## Technologies covered

| Technology | What it is | Module | Used in this portfolio? |
|---|---|---|---|
| RAG | retrieve passages, then answer from them with citations | all | yes |
| Semantic search | embedding-based retrieval by meaning | 02 | yes |
| LangChain | framework for loaders, splitters, vector stores and retrievers | 03 | yes (FAISS retriever) |
| LlamaIndex | a second RAG framework, index-centred | 03 | yes (structured specs) |
| LiteLLM | one interface to many model providers | 04 | yes (routes to Bedrock Nova) |
| Pydantic | typed schemas and validation | 04 | yes |
| LangGraph | agents as state machines | 05 | yes |
| Langfuse | LLM tracing and cost tracking | 09 | yes |
| LangSmith | LangChain's hosted tracing and evaluation | 09 | no (Langfuse already covered it), explained for comparison |
| GraphRAG, Neo4j | retrieval over a knowledge graph | 10 | no (the data is not graph-shaped), explained with a small runnable example |

The last two rows are deliberate: the course says plainly which tools were *not* needed and why, so you
can choose by the shape of your problem rather than by fashion.

## Data and licences

The Harbourview handbook and its questions are original text written for this course; they are covered by
the repository licence. The embedding model `all-MiniLM-L6-v2` is Apache 2.0.

## Cost and resources

Everything runs on a CPU. Modules 01 and 02 execute in under a minute each (the embedding model load is
about 2 s once downloaded). Later modules that show a real hosted model do so with clearly marked optional
cells; the executed results never depend on an API key.
