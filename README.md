# Florence-2 Component-Level Profiler

Instrumentation that breaks a **Florence-2** inference pass into its individual pipeline stages and reports per-component timing and peak memory. Built to find which stage actually dominates latency on constrained hardware, rather than treating the model as a black box.

## What it measures

The profiler wraps the model and records wall-clock time for each stage separately:

- Image loading and preprocessing
- Vision patch embedding and **vision encoder** (with optional per-stage breakdown)
- Position / temporal embeddings and projection
- Text tokenization and embedding, embedding merge
- Encoder forward, decoder forward
- **Prefill vs. decode** split, first-token latency, average per-token time
- Post-processing, total time, peak memory

Results are written to both `florence2_profiling.json` and `florence2_profiling.csv` (one row per run) for direct comparison across configurations.

## Key result

On the baseline FP32 CPU configuration, the **vision encoder is the single largest cost** — 4.77 s of a 13.1 s total — with projection (2.04 s), encoder forward (3.67 s), and decoder forward (5.10 s) following. Prefill accounts for 10.3 s against 2.8 s of decode, so the first token dominates end-to-end latency for short generations.

## Files

| File | Purpose |
|---|---|
| `florence2_profiler.py` | Core profiler; `ProfilingMetrics` dataclass and stage instrumentation |
| `modeling_florence2_profiled.py` | Florence-2 modeling code with timing hooks inserted |
| `benchmark_all.py` | Runs the full benchmark sweep across configurations |
| `compare_results.py` | Compares runs and emits `benchmark_comparison.png` |
| `optimizations/` | Baseline and quantization variants (`baseline.py`, `opt1_quantization.py`) |
| `app.py` | Streamlit interface for interactive runs |
| `10-florence2_test.ipynb`, `20-florence_2.ipynb` | Exploratory notebooks |
| `images/` | Test image set used for the benchmarks |

## Usage

```bash
pip install -r requirements-florence.txt
python benchmark_all.py        # run the sweep
python compare_results.py      # generate comparison plot
```
