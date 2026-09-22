Contains code for the evaluations used in the paper.


## Setup

`evals/`, `prompt_util/`, `eval_main.py`, `eval_layercast.py` and `patch_vllm.py` are
derived from  LayerCast (https://github.com/nanomaoli/llm_reproducibility). `bootstrap.sh` fetches that repo and applies `upstream.patch`, which holds the modifications made for this work.

```bash
./bootstrap.sh
```


## Environment

Two conda envs, both python 3.12. Exact versions are pinned in the two
requirements files.

**`rf`** everything except the figures (CUDA 12.4 GPU required)

```bash
conda create -n rf python=3.12 -y
conda activate rf
pip install -r requirements.txt
```

**`rf-plot`** the `plot/` figure scripts only; no GPU, no vLLM

```bash
conda create -n rf-plot python=3.12 -y
conda activate rf-plot
pip install -r requirements-plot.txt
```


## 1. Kernel and integration

|   |   |
|---|---|
| `rf_kernels.py` | The RF GEMM. Fused BF16-load + in-register FP32 upcast, IEEE FMA on every `tl.dot`, pinned compile-time configs chosen by a function of `(M,N,K)`, deterministic split-K with contiguous segments and an ordered second-pass reduce, segmentation independent of M within the decode bucket. |
| `patch_vllm.py` | vLLM patches for Qwen2 / Qwen3 / Llama: BF16 weight storage, FP32 RoPE / attention / KV cache. Set `RF_FUSED` to 0 for LayerCast baseline and 1 (default) for RF. `RF_BF16_LMHEAD=1` (default) stores `lm_head` in BF16 on untied models. |
| `sweeps/tune_sweep.py` | Development tool used to pick the two pinned configs in `rf_kernels.py` |


## 2. Bitwise probes (cross-architecture equality)

|   |   |
|---|---|
| `probe_cross_arch.py` | RF bitwise equality across A100 / L40S / H100. |
| `probe_baseline_claims.py` | Check LayerCast cast-then-cuBLAS is not bitwise equal across architectures, and cuBLAS row 0 is batch-variant while RF is not. |

Run `dump` on each GPU into a shared directory, then `compare` once:

```bash
python probe_cross_arch.py dump        # on each GPU
python probe_cross_arch.py compare
python probe_baseline_claims.py dump   # on each GPU
python probe_baseline_claims.py compare
```


## 3. End-to-end evaluation

|   |   |
|---|---|
| `eval_main.py` | Unmitigated BF16 baseline using `--dtype bfloat16`. |
| `eval_layercast.py` | LayerCast and RF. `--dtype float32` required. `RF_FUSED` env var selects which. |
| `evals/`, `prompt_util/` | Benchmark harness, forked from SkyThought. Task configs and scorers. |

Both write `outputs/<mode>/<exp_name>/<model>/` with per-problem `.pt` tensors of
top-5 token IDs and logprobs, and score in-process into `scoring_results/`.

```bash
python eval_main.py      --model <M> --task math500 --dtype bfloat16 \
                         --seed 42 --batch_size 32 --max_tokens 4096 \
                         --exp_name timing-a100_math500_timing_bf16-42

RF_FUSED=1 python eval_layercast.py --model <M> --task math500 \
                         --dtype float32 --seed 42 --batch_size 32 \
                         --max_tokens 4096 \
                         --exp_name timing-a100_math500_timing_rf-42
```

`divergence_analysis.py` parses `<gpu>`, `<task>`, `<method>` and `<seed>` back out of
the `exp_name`, so keep the `timing-<gpu>_<task>_timing_<method>-<seed>` form if
the run is to be included in the divergence analysis.

|   |   |
|---|---|
| `sweeps/run_sweep.py` | Runs the paper grid (method x seed x model x task) and times every run. |

```bash
python sweeps/run_sweep.py                    # all 3 methods, 3 seeds, 3 models, 4 tasks
python sweeps/run_sweep.py --methods bf16     # unmitigated BF16 baseline only
python sweeps/run_sweep.py --mode memory      # only memory profiling
python sweeps/run_sweep.py --tasks gsm8k --seeds 42 --dry-run
```

`--methods`, `--models`, `--tasks`, `--seeds`, `--batch-size` and `--max-tokens` narrow the
grid. `--gpu` overrides detection. `--tag` labels the logs. `--no-resume` applies to
the BF16 runs (`eval_layercast.py` has no such flag). `--mode memory` sweeps
`profile_memory.py` over both methods and logs to `<stamp>-<gpu>-memory-times.log`.

|   |   |
|---|---|
| `smoke_e2e.py` | Short greedy generation through patched vLLM to check that the two paths load and decode. |


## 4. Analysis

|   |   |
|---|---|
| `divergence_analysis.py` | Token-level cross-GPU and cross-seed divergence from the stored `.pt` token streams. Writes `divergence_summary.csv` and `divergence_firsts.csv`. |
| `logit_analysis.py` | Greedy decision margins vs. the cross-architecture perturbation of that margin, in nats. Writes `logit_hist.npz`, `logit_noise_summary.csv`, `logit_divpoint.csv`, `logit_gap_summary.csv`. |
| `logit_gap_analysis.py` | Earlier per-model and per-task gap-distribution analysis. Writes `logit_flip_prediction.csv`. |
| `timing_to_csv.py` | Parses the `*-times.log` files from root directory into `<gpu>-timing-comparison.csv`. |
| `profile_memory.py` | Resident weights, peak forward working set, and KV blocks at fixed `gpu_memory_utilization`. |


## 5. Ablation of design rules R1-R4

|   |   |
|---|---|
| `ablation/ablation_kernels.py` | A copy of `rf_kernels.py` with each rule turned into a knob. |
| `ablation/check_parity.py` | Asserts `ablation_kernels.py` is identical to the shipped kernel with all rules on. |
| `ablation/a1_precision.py` | R1: IEEE vs TF32 vs TF32x3 |
| `ablation/a2_autotune.py` | R2: autotuner search |
| `ablation/a3_splitk.py` | R3: none vs ordered vs atomic split-K |
| `ablation/a4_batch_invariance.py` | R4: M sweep |

```bash
cd ablation && ./run_ablations.sh    # on each of A100, L40S, H100
python analyze.py                    # cross-architecture report and tables
```


## 6. Figures

|   |   |
|---|---|
| `plot/paper_style.py` | Shared matplotlib style. |
| `plot/plot_timing_comparison.py` | End-to-end time bars per GPU x model x benchmark. |
| `plot/plot_divergence_bars.py` | Cross-GPU divergence rate per method. |
| `plot/plot_divergence_cdf.py` | Cumulative divergence vs. token position. |
| `plot/plot_logit_gap.py` | Decision margins vs. cross-architecture noise. |
| `plot/plot_speedup_roofline.py` | Speedup vs. GPU FP32-FLOP-per-byte ratio. |

```bash
python plot/plot_<name>.py
```
