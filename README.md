# Thermo Pulse

**Entropy-feedback temperature control for autoregressive language model decoding**

Thermo Pulse is an experimental, inference-time decoding controller that adjusts token-sampling temperature using the *level* and *step-to-step change* of model-output entropy. It is motivated by the mathematical relationship between temperature-scaled softmax and the Boltzmann distribution, and is implemented as a discrete-time proportional–derivative (PD) feedback rule.

This repository contains the **V14 Mechanism & Utility Benchmark**: nine decoding configurations (M1–M9), deterministic synthetic logic and mathematics question sets, a Qwen2.5-3B-Instruct evaluation script, per-token mechanism diagnostics, and the recorded V14 results. It also includes a separate `m9_local_engine.py` module for interactive generation integrations.

> **Research status:** Exploratory single-model study. The V14 results do **not** establish statistically significant overall superiority of Thermo Pulse, prove that an entropy drop predicts a repetition loop, or demonstrate a physical phase transition in language models.

## Motivation

Fixed-temperature sampling uses one sampling temperature throughout generation, even though the token probability distribution changes across steps. Thermo Pulse treats uncertainty as an observable feedback signal rather than assuming one temperature is appropriate for every state.

The physical analogy is limited: the connection to statistical mechanics is the **mathematical form of the distribution**, not evidence of Ising-type behavior, thermodynamic phase transitions, or a universal critical temperature in an LLM.

## Controller

At decoding step `t`, compute the model's unscaled next-token probabilities, take the **top five**, renormalize their probabilities, and calculate truncated Shannon entropy:

\[
\widehat p_{t,i} = \frac{p_{t,i}}{\sum_{j\in\mathrm{Top5}}p_{t,j}},\qquad
H_t = -\sum_{i\in\mathrm{Top5}}\widehat p_{t,i}\ln\widehat p_{t,i}.
\]

Let `ΔH_t = H_t − H_(t−1)` (zero for the initial step). The **M8 Full PD** controller uses

\[
T_t = \operatorname{clip}\!\left(
0.50 + 0.25(H_t - 1.20) - 0.15\Delta H_t,\;0.01,\;1.20
\right).
\]

The controller applies `softmax(logits / T_t)` at the **same** decoding step. The **M9 Adaptive Hybrid** adds a relative probability cutoff: after temperature scaling, remove candidate tokens whose probability is less than **5% of the highest-probability candidate**, then renormalize the remaining distribution. The baseline coefficients and cutoff were frozen for the V14 comparison.

**Interpretation:** the proportional term reacts to current uncertainty; the derivative term reacts to its change. Whether these signals produce better answers is an *empirical question*, not guaranteed by the controller design.

## Evaluated decoding modes

| Mode | Configuration | Purpose |
|---|---|---|
| M1 | Greedy | Deterministic reference |
| M2 | Fixed temperature `T = 0.2` | Low-temperature reference |
| M3 | Fixed temperature `T = 0.7` | Standard sampling reference |
| M4 | Entropy-threshold temperature | Discrete entropy-based reference |
| M5 | Mirostat v2 variant | Surprise-feedback reference |
| M6 | Thermo Pulse P-only | Ablation: entropy level |
| M7 | Thermo Pulse D-only | Ablation: entropy change |
| M8 | Thermo Pulse Full PD | Combined feedback controller |
| M9 | Thermo Pulse Adaptive Hybrid | Full PD plus 5% relative-probability cutoff |

M5 refers to the **implementation in `benchz.py`**; results should not automatically be generalized to every other implementation or parameterization of Mirostat.

## V14 experimental protocol

| Property | V14 configuration |
|---|---|
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Domains | Logic and mathematics |
| Questions | 200 unique questions per domain (400 total) |
| Decoding modes | 9 (`M1`–`M9`) |
| Random seeds | 5 per question and mode |
| Total generations | 400 × 9 × 5 = **18,000** |
| Generation limit | **256 new tokens** |
| Dataset seed | `20261031` |
| GPU used for the reported run | NVIDIA GeForce RTX 4070 Laptop GPU |

The question sets are **synthetically generated, parameterized tasks** with programmatically computed answer keys. Each domain has 20 labeled families with 10 variants per family and four descriptive mechanism groups (`control`, `boundary`, `transition`, `recovery`). These labels are **not independently validated indicators of reasoning difficulty or of actual error recovery**. Several families reuse the same underlying problem-generation patterns, so questions are not all independent reasoning templates.

The V14 benchmark is **not GSM8K or the public MATH benchmark**; the included `benchmark_v14_*.json` files are the actual evaluation datasets.

## V14 results

**Mean correctness and mean generated tokens per response**, aggregated over the five seeds:

| Mode | Logic accuracy | Logic tokens | Math accuracy | Math tokens |
|---|---:|---:|---:|---:|
| M1 Greedy | 24.0% | 59.4 | **33.0%** | 79.8 |
| M2 Fixed low | 25.2% | 59.7 | 31.2% | 76.3 |
| M3 Fixed standard | 27.6% | **59.3** | 25.4% | **73.4** |
| M4 Entropy threshold | 25.6% | 59.4 | 29.2% | 75.8 |
| M5 Mirostat v2 | 27.6% | 59.4 | 25.4% | 73.6 |
| M6 P-only | 25.3% | **59.3** | 26.9% | 76.5 |
| M7 D-only | 26.0% | **59.3** | 28.8% | 75.1 |
| M8 Full PD | 26.7% | 59.8 | 28.6% | 76.2 |
| **M9 Adaptive Hybrid** | **27.7%** | 59.8 | 29.5% | 76.4 |

All numbers above are descriptive experimental measurements from the committed `qwen3b_v14_results_s200.json`. Bold values mark column maxima or minima, **not statistical significance**.

In particular:

- M9 exceeded M8 by **1.0 percentage point** on logic and **0.9 percentage point** on mathematics.
- Compared with M3, M9 differed by **+0.1 pp** on logic and **+4.1 pp** on mathematics.
- Greedy achieved the highest mathematics accuracy (**33.0%**) in this experiment.
- No blanket claim that M9 is the most accurate or most token-efficient mode is supported.

### Statistical interpretation

The study uses paired comparisons on the same questions and five seeds, with sensitivity analyses clustered by the 20 labeled problem families. The four designated M9 comparisons against M3 and M8 have **family-cluster bootstrap 95% confidence intervals that include zero**; no statistically secure superiority is established by these primary tests. Seeds and multiple instances from the same family must not be counted as independent problem families.

Relevant outputs include `paired_comparisons`, `mcnemar`, and per-item arrays within the results JSON. Exploratory groupwise differences are available in the contrasts CSV; do not treat them as pre-registered causal effects.

### Mechanism and token-efficiency metrics

The JSON and HTML report include generation diagnostics such as observed entropy, temperature, candidate filtering, and token counts. The M9 diagnostic averages for V14 were approximately:

| Measurement | Logic | Mathematics |
|---|---:|---:|
| Mean applied temperature | 0.2684 | 0.2733 |
| Mean surviving candidates | 1.23 | 1.25 |
| Fraction of steps with filtering | 49.72% | 48.30% |
| Mean removed probability mass | 0.3544% | 0.4137% |

Removing many low-probability candidates does **not** necessarily remove much *probability mass*, nor does it by itself establish better reasoning accuracy. The manuscript interprets these measurements as controller behavior rather than proof of a causal accuracy benefit.

For cost comparisons, prefer **total generated tokens divided by the number of correct generations** over the mean token length of correct generations alone. Both correctness and output length affect cost per successful answer.

## Repository contents

```text
thermo_pulse/
├── README.md                              # This document
├── benchz.cmd                             # Windows CUDA benchmark launcher
├── benchz.py                              # M1–M9 decoding benchmark
├── benchz_dataset_v14.py                  # Deterministic V14 dataset generator/loader
├── benchmark_v14_logic_mechanism.json     # 200 logic questions and answer keys
├── benchmark_v14_math_mechanism.json      # 200 mathematics questions and answer keys
├── post_v14.py                            # Mechanism-group and paired-contrast exports
├── m9_local_engine.py                     # Separate interactive-generation engine module
├── requirements.txt                       # Existing general-purpose dependencies
├── qwen3b_v14_results_s200.json           # Full recorded V14 results
├── qwen3b_v14_report_s200.html            # Token-level HTML visualization
├── qwen3b_v14_results_s200_mechanisms.csv # Descriptive mechanism-group results
└── qwen3b_v14_results_s200_contrasts.csv  # Descriptive paired comparisons
```

**Scope note:** This archive does not contain `thermo_server.py`, `tpchat.html`, `tpchat.cmd`, or V15 fork scripts. `m9_local_engine.py` is a reusable module, **not a standalone packaged web-chat server**. Its default model is `Qwen/Qwen3.5-2B`, which is different from the Qwen2.5-3B model used in the V14 benchmark.

## Run the benchmark (Windows + NVIDIA CUDA)

### Requirements

- Windows with Python **3.10**, accessible as `py -3.10`.
- An NVIDIA GPU with working CUDA drivers and sufficient VRAM for the selected model.
- Network access to PyPI, the PyTorch CUDA wheel index, and Hugging Face for the initial installation/model download.
- Enough available disk space for PyTorch, the model weights, and recorded outputs.

The committed `benchz.cmd` creates or reuses an isolated `.venv_benchz_gpu` environment. It installs **PyTorch 2.9.1 CUDA 12.6 wheels**, then installs the benchmark packages if missing. It intentionally stops if CUDA is unavailable rather than silently falling back to CPU.

> **Dependency caveat:** The included `requirements.txt` is a legacy/general dependency list and does not fully specify the V14 benchmark's dependencies. For reproducing the reported Windows GPU experiment, use the environment setup in **`benchz.cmd`** rather than assuming `pip install -r requirements.txt` alone is sufficient.

### Run the full V14 benchmark

From Command Prompt in the extracted repository directory:

```bat
benchz.cmd
```

The launcher performs the full benchmark (200 questions per domain, five seeds, nine modes) and then runs `post_v14.py`. The smoke/pilot commands remain commented out in the batch file.

The launcher calls an equivalent benchmark command:

```bat
set BENCH_MAX_NEW_TOKENS=256
set BENCH_REQUIRE_FULL_CUDA=1
.venv_benchz_gpu\Scripts\python.exe benchz.py --model qwen-3b --macro --sample 200 --seeds 5 --start M1 --output qwen3b_v14_results_s200.json --html_output qwen3b_v14_report_s200.html
.venv_benchz_gpu\Scripts\python.exe post_v14.py qwen3b_v14_results_s200.json
```

**Warning:** Running the batch file with the filenames above will **overwrite existing result/report files**. Back up the included V14 artifacts or change the output paths before reproducing the experiment.

### Generate analyses from the recorded results without a GPU

With Python and NumPy installed:

```bash
python post_v14.py qwen3b_v14_results_s200.json
```

This recreates the two CSV exports from the stored results and does not rerun the model.

### Inspect the results

Open `qwen3b_v14_report_s200.html` locally in a browser, or inspect the raw JSON and CSV files programmatically. The HTML is a **sample-stream visualization**, not a replacement for the complete per-item results in JSON.

## Reproducibility and limitations

1. **Single model and hardware configuration.** No cross-model generalization or latency/throughput improvement is established.
2. **Synthetic benchmark.** Generated questions are objectively keyed but are not an independently validated external benchmark. The four mechanism labels describe dataset construction, not verified cognitive processes.
3. **Controller coefficients were fixed.** This benchmark does not establish that the chosen gains or 5% threshold are globally optimal.
4. **Statistical dependence.** Five seeds and parameter variants from the same family do not constitute fully independent task families. Family-level confidence intervals are sensitivity analyses on a limited family set.
5. **Accuracy is distinct from mechanism.** Evidence that a filter activates or temperature changes is not evidence that those changes caused a correct answer.
6. **Generation length is not latency.** Average generated tokens and tokens per correct answer are not direct throughput, power, or serving-cost measurements.
7. **Scientific framing.** Boltzmann/thermodynamic language is used as a mathematical analogy. The experiment does not test physical phase transitions, Ising models, or a repetition-loop “circuit breaker.”

## Research context and references

- Vaswani et al. (2017), *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Holtzman et al. (2020), *The Curious Case of Neural Text Degeneration*. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
- Basu et al. (2021), *Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity*. [arXiv:2007.14966](https://arxiv.org/abs/2007.14966)
- Wei et al. (2022), *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- Yang et al. (2024), *Qwen2.5 Technical Report*. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)

These sources provide background; they should not be interpreted as independent validation of Thermo Pulse. See the accompanying research manuscript for the broader discussion and methodology.

## Citation, license, and release notes

**Citation:** A preprint DOI or repository citation has not been supplied. Please add the final bibliographic entry and a `CITATION.cff` file after publication; do not cite a placeholder DOI.

**License:** No license file was included in the supplied archive. Until the repository owner selects and adds a license, public availability on GitHub should **not** be interpreted as granting reuse or redistribution rights.

**Suggested Git hygiene:** Exclude `.venv_benchz_gpu/`, `__pycache__/`, downloaded model caches, private credentials, and local machine settings from version control. Preserve the recorded result artifacts if the intention is to publish a reproducibility package.
