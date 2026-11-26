# Neural Networks 1002: Advanced and Current Topics

Sixteen hands-on Jupyter notebooks on the topics that follow the basics: how language models are aligned and fine-tuned, what makes architectures and training efficient, how images, sound and speech are generated, how neural networks handle graphs and physics, how to explain, calibrate, audit and secure a model, how to fix the data instead of the model, how agents use tools, and where generalist models are heading. The last module is a narrative case study on real wind-energy problems, with five bundled notebooks. Every idea is built at a scale a laptop CPU can train, measured against an honest baseline, and compared with a real pre-trained model wherever one is small enough to run. No GPU and no API keys are needed.

**Prerequisite:** this is the sequel to the introductory neural-networks course, the basics and overview ([neural-networks.academy.forwardforecasting.eu](https://neural-networks.academy.forwardforecasting.eu/), source in [`../neural-networks/`](../neural-networks/)): neurons, backpropagation, training well, reinforcement learning, convolutions, recurrent networks, the transformer and a GPT. Each notebook here says which of those modules it builds on, with links. If you know PyTorch tensors, a training loop and the transformer, you can start here.

**Course website:** [neural-networks-1002.academy.forwardforecasting.eu](https://neural-networks-1002.academy.forwardforecasting.eu/) renders every notebook with its saved outputs, a syllabus, progress tracking, and download / Open-in-Colab links.

**Versión en español:** the whole course is also available in Spanish, notebooks included, at [neural-networks-1002.academy.forwardforecasting.eu/es/](https://neural-networks-1002.academy.forwardforecasting.eu/es/) and in the [`es/`](es/) folder.

## Table of Contents

- [The notebooks](#the-notebooks)
- [Learning path](#learning-path)
- [Data flow](#data-flow)
- [Libraries and techniques explained](#libraries-and-techniques-explained)
- [Why these models and datasets](#why-these-models-and-datasets)
- [Getting started](#getting-started)
- [Runtime and cost](#runtime-and-cost)
- [References](#references)
- [Contact](#contact)

## The notebooks

CPU time is the measured run time of the whole notebook on a free Google Colab runtime (2 virtual CPUs, no GPU), rounded, without the one-off downloads; your laptop may be faster or slower.

| # | Notebook | What you build | Key ideas | CPU time |
|---|---|---|---|---|
| 01 | [`01_rlhf_and_alignment.ipynb`](01_rlhf_and_alignment.ipynb) | The whole post-training pipeline at toy scale: a tiny GPT, supervised fine-tuning, a reward model, policy-gradient fine-tuning, DPO and GRPO | Bradley-Terry reward model, KL penalty, PPO clip, reward over-optimisation (Goodhart's law), DPO, GRPO with a verifiable reward | ~2 min |
| 02 | [`02_open_models_and_fine_tuning.ipynb`](02_open_models_and_fine_tuning.ipynb) | Real pre-trained checkpoints loaded on a laptop and fine-tuned, LoRA from scratch and with `peft`, models saved and exported | open source vs open weights, licences, quantisation, linear probe vs partial vs full fine-tuning, LoRA / QLoRA, `state_dict` vs pickle, safetensors, ONNX, model cards | ~4 min |
| 03 | [`03_efficient_architectures_and_scale.ipynb`](03_efficient_architectures_and_scale.ipynb) | A cost calculator, tiled FlashAttention, grouped-query attention, a mixture of experts, a state space model, and data parallelism on several CPU processes | the 6ND cost rule, KV cache size, RoPE, sliding windows, expert collapse, S4 and Mamba, mixed precision, gradient checkpointing, ZeRO/FSDP, speculative decoding, energy and cost | ~8 min |
| 04 | [`04_autoencoders_gans_and_self_supervised.ipynb`](04_autoencoders_gans_and_self_supervised.ipynb) | An autoencoder, a VAE, a GAN, a SimCLR-style learner and a CLIP in miniature, next to a real CLIP | bottleneck and anomaly detection, ELBO and reparameterisation, mode collapse, InfoNCE, linear-probe curves, zero-shot classification | ~1 min |
| 05 | [`05_diffusion_models.ipynb`](05_diffusion_models.ipynb) | A diffusion model from scratch on digits, with faster sampling, guidance and a latent version | forward noising, noise schedules, the denoising U-Net, DDPM vs DDIM, classifier-free guidance, latent diffusion, how text-to-image is assembled | ~3 min |
| 06 | [`06_audio_and_video_generation.ipynb`](06_audio_and_video_generation.ipynb) | Spectrogram and codec pipelines, token and diffusion models for sound, a real music model, a video diffusion toy | STFT and mel spectrograms, EnCodec, Griffin-Lim vs HiFi-GAN, Fréchet distance, MusicGen, temporal consistency, optical flow, watermarking | ~17 min |
| 07 | [`07_speech_recognition_synthesis_and_voice.ipynb`](07_speech_recognition_synthesis_and_voice.ipynb) | A CTC speech recogniser from scratch, then real Whisper, wav2vec 2.0 and text-to-speech models measured | CTC loss, greedy and beam decoding, word error rate, Whisper hallucination on silence, round-trip WER, translation cascade, speaker embeddings, voice cloning, deepfakes and consent | ~12 min |
| 08 | [`08_graph_neural_networks.ipynb`](08_graph_neural_networks.ipynb) | GCN, GraphSAGE, GAT and GIN layers in pure PyTorch, on a generated network and on real molecules | message passing, permutation invariance, over-smoothing and over-squashing, readout, cross-validation, an honest result against simple baselines | ~1 min |
| 09 | [`09_scientific_machine_learning.ipynb`](09_scientific_machine_learning.ipynb) | A finite-difference solver, PINNs for an ODE and the heat equation, an inverse problem, a DeepONet and a Fourier neural operator | physics in the loss, loss weighting, hard constraints, spectral bias and Fourier features, causal training, operator learning, speed-ups, large scientific models | ~9 min |
| 10 | [`10_interpretability_and_explainability.ipynb`](10_interpretability_and_explainability.ipynb) | Attribution methods on tabular, vision and language models, each checked against a known truth, plus a first look at mechanistic interpretability | permutation importance, integrated gradients, Shapley values, LIME, Grad-CAM, the randomisation sanity check, deletion curves, probes, induction heads, activation patching, sparse autoencoders | ~3 min |
| 11 | [`11_uncertainty_calibration_and_robustness.ipynb`](11_uncertainty_calibration_and_robustness.ipynb) | Calibrated, uncertainty-aware and attack-tested classifiers | aleatoric vs epistemic, ECE, temperature scaling, deep ensembles, MC dropout, Laplace, conformal prediction, quantile regression, FGSM and PGD, adversarial training | ~25 s |
| 12 | [`12_responsible_ai.ipynb`](12_responsible_ai.ipynb) | Fairness, privacy and safety measurements on census data and small language models | demographic parity, equalised odds, the impossibility result, reweighing, membership inference, DP-SGD, red teaming, text watermarking, model cards, EU AI Act and NIST overview, energy | ~5 min |
| 13 | [`13_data_centric_ai.ipynb`](13_data_centric_ai.ipynb) | Label-noise, leakage, valuation and active-learning experiments with the model held fixed | cost of noisy labels, confident learning, Cohen's kappa, duplicates and leakage, learning curves, Shapley data values, active learning, weak supervision, model collapse, data monitoring | ~5 min |
| 14 | [`14_agents_and_orchestration.ipynb`](14_agents_and_orchestration.ipynb) | Agents with tools, memory and planning around a simulated model, and seven orchestration patterns measured on the same tasks | function calling, ReAct, guardrails, router, supervisor, hierarchical planner, critic loop, pass@k vs pass^k, MCP, prompt injection | ~10 s |
| 15 | [`15_generalist_models_and_agi.ipynb`](15_generalist_models_and_agi.ipynb) | One small transformer that reads images and words in a single token stream, and a scaling experiment with a tested extrapolation | multimodal tokens, scaling laws, emergence, chain of thought, test-time compute, world models, what AGI means and why it is hard to measure | ~2 min |
| 16 | [`16_real_use_cases_wind_energy.ipynb`](16_real_use_cases_wind_energy.ipynb) | **Narrative capstone:** five real wind-energy problems, each testing one technology against a classical method, with every number checked by code | PINN for the vertical wind profile, GNN for wakes, spectral wave rose, XGBoost vs PyTorch, CNN on radar images, negative results, data licences | ~5 s (no new experiments) |

Each notebook ends with key takeaways and exercises. The notebooks are saved **with their outputs**, so you can read them on GitHub, on the course website, or in any notebook viewer without running anything.

**Module 16 and `wind_cases/`.** The folder [`wind_cases/`](wind_cases/) holds five Spanish notebooks (with a Spanish decimal comma) that were executed beforehand on real open wind data, with the small data files in `wind_cases/datos/`. Module 16 is the English narrative over them and checks every figure it quotes. Some of that data is licensed for non-commercial use only; read the module and [`wind_cases/README.md`](wind_cases/README.md) before reusing or publishing anything.

## Learning path

```
 Part 1 · Adapting and aligning       Part 2 · Representations and generation      Part 3 · Other architectures
 01 RLHF, DPO, GRPO                   04 Autoencoders, VAEs, GANs, CLIP            08 Graph neural networks
   ▼                                    ▼                                            ▼
 02 Open models, LoRA, export         05 Diffusion models                          09 Scientific ML: PINNs, operators
   ▼                                    ▼
 03 Efficient architectures, scale    06 Audio and video generation
                                        ▼
                                      07 Speech: ASR, TTS, voice cloning

 Part 4 · Reliable and responsible AI          Part 5 · Systems and the horizon     Part 6 · Capstone
 10 Interpretability and explainability        14 Agents and orchestration          16 Real use cases in wind energy
   ▼                                             ▼                                     (points back to 08, 09, 11)
 11 Uncertainty, calibration, robustness       15 Generalist models and AGI
   ▼
 12 Responsible AI: fairness, privacy, safety
   ▼
 13 Data-centric AI
```

The parts are mostly independent: after the prerequisite course you can take Part 2, 3 or 4 in any order. Inside a part, go in order. Module 06 assumes 04 and 05, module 07 loosely assumes 06, module 12 assumes 10 and 11, and module 16 refers back to 08, 09 and 11.

## Data flow

Every notebook follows the same loop: take data, train or load a model, measure it against a baseline, and look inside or stress it.

```mermaid
flowchart LR
    subgraph Data["Data (mostly generated or bundled)"]
        D1[toy text and sorting task<br/>nb 01, 15]
        D2[sklearn digits 8x8<br/>nb 04, 05, 11, 13, 15]
        D3[public text and image models<br/>Hugging Face hub, cached once<br/>nb 02, 03, 04, 06, 07, 10]
        D4[small open datasets<br/>MUTAG, UCI Adult, Banking77,<br/>spoken digits, LibriSpeech samples<br/>nb 02, 07, 08, 10, 12]
        D5[simulated physics and tasks<br/>nb 09, 14]
        D6[open wind-farm and wave data<br/>wind_cases, nb 16]
    end
    Data --> M[Train from scratch<br/>or load a pre-trained checkpoint]
    M --> A[Adapt or generate<br/>fine-tune, align, sample, denoise]
    A --> E[Measure against a baseline<br/>held-out data, fixed seeds]
    E --> X[Look inside and stress it<br/>attribution, calibration, attacks,<br/>fairness, privacy]
    X --> R[Report honestly<br/>including negative results]
```

## Libraries and techniques explained

| Library | Used for | Why |
|---|---|---|
| **PyTorch** (CPU build) | Every notebook | Automatic differentiation, ready-made layers, and the same API used to train real large models; `torch.distributed` for real data parallelism on CPU processes in module 03 |
| **NumPy**, **SciPy** | Numerics, solvers, signal processing, statistics | The classical baselines (finite differences, spectra, optimisation) every neural method is compared with |
| **scikit-learn** | Bundled digits, baselines, splits, metrics | Small offline datasets and standard evaluation |
| **transformers**, **huggingface_hub**, **safetensors** | Loading real pre-trained models: BERT-Mini, DistilBERT, CLIP, Whisper, wav2vec 2.0, SpeechT5, a small mixture-of-experts model, a small Mamba model | Real checkpoints show what the toy models cannot; everything is cached after the first download |
| **peft** | LoRA in module 02 | The reference implementation to check our from-scratch LoRA against |
| **ONNX**, **ONNX Runtime**, **onnxscript** | Exporting a model and checking it runs outside PyTorch | The standard interchange format |
| **diffusers** (optional) | A real compressed text-to-image pipeline in module 05 | Shows how the from-scratch parts are assembled in a real system |
| **soundfile**, **speechbrain**, **sentencepiece**, **sacremoses** | Audio files, speaker embeddings, tokenisers for the speech and translation models | The speech notebooks need real audio tooling |
| **pandas**, **pyarrow**, **scikit-image**, **XGBoost** | Tables, the bundled sample photographs, and the gradient-boosting baseline | Honest baselines and small real data |
| **SHAP**, **cleanlab**, **Opacus** (optional) | Cross-checks of our own implementations in modules 10, 13 and 12 | A method we wrote ourselves is only trusted once it matches the reference library |
| **Matplotlib** | All plots | Standard, renders inline in the notebook |
| **Jupyter** + **nbconvert** | The notebook format, and rendering the course website | Mixes explanation, maths (LaTeX), code and output in one document |

Key techniques, and where each one is introduced:

- **Post-training** (01): supervised fine-tuning, reward models, KL-regularised policy gradient, DPO, GRPO.
- **Using open checkpoints** (02): licences, quantisation, parameter-efficient fine-tuning, safe saving and export.
- **Cost of scale** (03): FLOP and memory accounting, FlashAttention, GQA, mixture of experts, state space models, distributed training, energy.
- **Learning without labels** (04): autoencoders, VAEs, GANs, contrastive learning and CLIP.
- **Generative models** (04, 05, 06, 07): diffusion, discrete audio tokens, vocoders, video consistency, recognition and synthesis of speech.
- **Structure beyond grids** (08, 09): message passing on graphs, physics-informed networks and neural operators.
- **Trust** (10, 11, 12, 13): explanation methods that are tested, calibration and conformal prediction, adversarial robustness, fairness, differential privacy, red teaming, label and data quality.
- **Systems** (14, 15): tool-using agents and their orchestration patterns, one model for many modalities, scaling laws.
- **Applying it** (16): when a neural network beat a classical method on real data, and when it did not.

## Why these models and datasets

- **Small on purpose, real where it fits.** The mechanisms (a reward model, a denoiser, a message-passing layer, a PINN, a CTC loss) are trained from scratch on toy problems so you see every step. Where a small real model exists (DistilBERT, Whisper tiny and base, CLIP, a 130 M-parameter Mamba), the notebook also runs it, so the toy and the real thing sit side by side.
- **Fixed seeds and baselines everywhere.** Every comparison includes the simple method (a linear model, a classical solver, XGBoost, random sampling) and says plainly when the fashionable method loses. Several notebooks report a negative result on purpose, for example graph networks against logistic regression on molecules.
- **Open data with licences stated.** Most notebooks use the UCI handwritten digits (Alpaydin and Kaynak, CC BY 4.0, bundled with scikit-learn). The others use small open sets credited where they are used: MUTAG molecules, UCI Adult (Becker and Kohavi, CC BY 4.0), Banking77 (CC BY 4.0), the Free Spoken Digit Dataset (CC BY-SA 4.0) and open wind-farm data on Zenodo (CC BY 4.0).
- **Licences are part of the lesson.** Some pre-trained models are non-commercial or gated (for example MusicGen weights, CC BY-NC 4.0), and some wind data is non-commercial. Each notebook says so next to the model or file, and module 02 teaches how to read a licence before building on a checkpoint.
- **Simulation where only simulation can answer.** The agents of module 14 run on a simulated model with documented, seeded error rates, because only a simulation lets us dial the errors and compare orchestration patterns fairly. The notebook says in its first lines that the results are about the architecture, not about any real model.
- **Honest scope.** The generalist-model module separates what has been measured and published from speculation, gives no dates and no probabilities for AGI, and labels speculation as such.
- **No sector in the way until the end.** Modules 01 to 15 use generic domains so the technology can be learned without domain knowledge. Module 16 then goes to a real sector, wind energy, and asks the practitioner's question: when did the neural network win, and when did a simpler method do better?

## Getting started

Requirements: Python 3.10+.

```bash
git clone https://github.com/fborbon/academy.git
cd academy/neural-networks-1002
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab        # or: jupyter notebook
```

The requirements file marks the heavier optional libraries (diffusers, speechbrain, SHAP, Opacus, cleanlab). A notebook that needs an optional library prints a clear message and skips that section if it is missing.

**A note on `torchaudio`.** While this course was written, torch 2.15 had no matching `torchaudio` wheel on the CPU index. SpeechBrain only needs the module to exist for inference, so module 07 installs a minimal stub when `torchaudio` is missing or mismatched. On a normal laptop with a matching torch and torchaudio pair, install both from the same index and the stub is not used.

Open the notebooks in the order of the learning path. To re-run one and refresh its saved outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace 05_diffusion_models.ipynb
```

Run notebooks one at a time: several PyTorch kernels in parallel compete for the same CPU threads and can each run many times slower.

**Downloads.** Several notebooks fetch pre-trained models or small datasets once, from the Hugging Face hub or the dataset's home, and cache them (the speech notebook caches about 3 GB; the mixture-of-experts and Mamba models in module 03 and the optional text-to-image model in module 05 are the next largest). Each notebook states what it will download in its first cell. Everything else works offline.

## Runtime and cost

Everything in the course runs locally on the CPU of an ordinary laptop. There are **no paid APIs**: no hosted language model, no API keys, no cloud training needed. Pre-trained models are downloaded once and run locally.

| Item | Cost |
|---|---|
| LLM / AI API calls | none ($0) |
| Compute to run the notebooks | your own machine: very roughly 4 to 5 hours to run all sixteen end to end on an 8-core laptop CPU (the speech, audio and video modules are the longest, 25 to 30 minutes each); the same again for the Spanish edition |
| Disk | about 15 MB of notebooks per language, plus the cached models and data (a few gigabytes in total if you run every notebook, mostly modules 03, 06 and 07) |

As an aside on how the course was produced, not something you need: the authors executed the heavier notebooks (and the Spanish editions) on an ephemeral 32-vCPU cloud instance that shut itself down when idle, to save waiting time; several timings in the notebooks come from that machine, and the table above scales them to a laptop. An instance of that size costs roughly 2 USD per hour, and the total for authoring and re-running both courses was on the order of tens of dollars, a rough estimate rather than an invoice. You can reproduce every result on a laptop, only more slowly.

## References

Each notebook ends with its own reference list with direct links; these are the anchor papers:

- Ouyang et al., [*Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155) (2022) · Rafailov et al., [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290) (2023) · Shao et al., [*DeepSeekMath (GRPO)*](https://arxiv.org/abs/2402.03300) (2024)
- Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685) (2021) · Dettmers et al., [*QLoRA*](https://arxiv.org/abs/2305.14314) (2023)
- Dao et al., [*FlashAttention*](https://arxiv.org/abs/2205.14135) (2022) · Shazeer et al., [*Outrageously Large Neural Networks (mixture of experts)*](https://arxiv.org/abs/1701.06538) (2017) · Gu and Dao, [*Mamba*](https://arxiv.org/abs/2312.00752) (2023)
- Kingma and Welling, [*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114) (2013) · Goodfellow et al., [*Generative Adversarial Nets*](https://arxiv.org/abs/1406.2661) (2014) · Chen et al., [*SimCLR*](https://arxiv.org/abs/2002.05709) (2020) · Radford et al., [*CLIP*](https://arxiv.org/abs/2103.00020) (2021)
- Ho, Jain and Abbeel, [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239) (2020) · Rombach et al., [*High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752) (2022)
- Défossez et al., [*High Fidelity Neural Audio Compression (EnCodec)*](https://arxiv.org/abs/2210.13438) (2022) · Copet et al., [*Simple and Controllable Music Generation*](https://arxiv.org/abs/2306.05284) (2023)
- Graves et al., [*Connectionist Temporal Classification*](https://doi.org/10.1145/1143844.1143891) (2006) · Radford et al., [*Whisper*](https://arxiv.org/abs/2212.04356) (2022) · Baevski et al., [*wav2vec 2.0*](https://arxiv.org/abs/2006.11477) (2020)
- Kipf and Welling, [*Semi-Supervised Classification with Graph Convolutional Networks*](https://arxiv.org/abs/1609.02907) (2017) · Veličković et al., [*Graph Attention Networks*](https://arxiv.org/abs/1710.10903) (2018) · Xu et al., [*How Powerful are Graph Neural Networks?*](https://arxiv.org/abs/1810.00826) (2019)
- Raissi, Perdikaris and Karniadakis, [*Physics-informed neural networks*](https://doi.org/10.1016/j.jcp.2018.10.045) (2019) · Lu et al., [*DeepONet*](https://arxiv.org/abs/1910.03193) (2021) · Li et al., [*Fourier Neural Operator*](https://arxiv.org/abs/2010.08895) (2021)
- Lundberg and Lee, [*A Unified Approach to Interpreting Model Predictions (SHAP)*](https://arxiv.org/abs/1705.07874) (2017) · Adebayo et al., [*Sanity Checks for Saliency Maps*](https://arxiv.org/abs/1810.03292) (2018) · Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (2021)
- Guo et al., [*On Calibration of Modern Neural Networks*](https://arxiv.org/abs/1706.04599) (2017) · Angelopoulos and Bates, [*A Gentle Introduction to Conformal Prediction*](https://arxiv.org/abs/2107.07511) (2021) · Goodfellow et al., [*Explaining and Harnessing Adversarial Examples*](https://arxiv.org/abs/1412.6572) (2015)
- Hardt, Price and Srebro, [*Equality of Opportunity in Supervised Learning*](https://arxiv.org/abs/1610.02413) (2016) · Abadi et al., [*Deep Learning with Differential Privacy*](https://arxiv.org/abs/1607.00133) (2016) · Mitchell et al., [*Model Cards for Model Reporting*](https://arxiv.org/abs/1810.03993) (2019)
- Northcutt et al., [*Confident Learning*](https://arxiv.org/abs/1911.00068) (2021) · Settles, [*Active Learning Literature Survey*](https://burrsettles.com/pub/settles.activelearning.pdf) (2009)
- Yao et al., [*ReAct*](https://arxiv.org/abs/2210.03629) (2022) · Kaplan et al., [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361) (2020)
- Datasets: Alpaydin and Kaynak, [*Optical Recognition of Handwritten Digits*](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) (UCI, CC BY 4.0) · Becker and Kohavi, [*Adult*](https://doi.org/10.24432/C5XW20) (UCI, CC BY 4.0) · Casanueva et al., [*Banking77*](https://arxiv.org/abs/2003.04807) (CC BY 4.0) · Jackson et al., [*Free Spoken Digit Dataset*](https://doi.org/10.5281/zenodo.1342401) (CC BY-SA 4.0) · Plumley, [*Penmanshiel and Kelmarsh wind-farm SCADA*](https://doi.org/10.5281/zenodo.5946808) (CC BY 4.0)

## Contact

Questions or suggestions: forwardforecasting@gmail.com
