# COCO Amplification

Run the COCO spatial-relation amplification experiment for supported
vision-language models.

This package reproduces the paper's "correcting incorrect predictions"
experiment on natural COCO scenes. Each example asks a two-option spatial
question about a subject object relative to a reference object, using
left/right or above/below relations. The script first measures the model's
strict baseline accuracy, then trains linear position probes from examples the
model answered correctly. It then amplifies the learned spatial directions in
the visual embeddings and checks how many original failures become correct.
A random-direction intervention is included as a control.

Supported models:

| Model flag | Hugging Face model | Notes |
|------------|--------------------|-------|
| `qwen` | `Qwen/Qwen2-VL-2B-Instruct` | Qwen COCO runner |
| `gemma` | `google/gemma-3-4b-it` | Gemma COCO runner; requires Hugging Face access |
| `pixtral` | `mistral-community/pixtral-12b` | Pixtral COCO runner; public model |

## Setup

From the repo root, first install the normal repo environment if you have not
already done so:

```bash
bash setup.sh
source .venv/bin/activate
```

For closest reproduction of the original COCO notebook results, use the
package-specific setup profile for the model you want to run:

```bash
bash coco_amplification/setup_repro.sh --model MODEL --fresh
source .venv/bin/activate
```

Use `qwen`, `gemma`, or `pixtral` for `MODEL`.

Without `--fresh`, the selected profile installs into the existing `.venv`.

Gemma is gated on Hugging Face. Before running Gemma, make sure your account
has access to `google/gemma-3-4b-it`. The runner will ask for a token if one is
not already available from `huggingface-cli login` or `HF_TOKEN`.

## Data

No manual data setup is required. The package already includes the small
two-object COCO QA file. On first run, it downloads the needed COCO validation
images and annotations automatically.

By default, downloaded COCO files are stored in:

```text
shared/coco_2obj/
```

To use a different location, pass:

```bash
python -m coco_amplification.run_experiment --coco-dir /path/to/coco_2obj
```

or set `COCO_DIR`.

## Run

Run the experiment:

```bash
python -m coco_amplification.run_experiment --model MODEL --recompute
```

Use `qwen`, `gemma`, or `pixtral` for `MODEL`.

To use compatible cached baseline and position-probe artifacts instead, omit
`--recompute`:

```bash
python -m coco_amplification.run_experiment --model MODEL
```

Run a small smoke test:

```bash
python -m coco_amplification.run_experiment --model MODEL --max-items 2 --alphas 0 1
```

Aggregate saved results:

```bash
python -m coco_amplification.plot
```

## Options

| Option | Description |
|--------|-------------|
| `--model MODEL` | Model to run. Use `qwen`, `gemma`, or `pixtral`. |
| `--torch-dtype DTYPE` | Precision. Use `fp32` or `bf16`; default is `fp32`. |
| `--coco-dir PATH` | COCO cache directory. Default is `$COCO_DIR` or `shared/coco_2obj/`. |
| `--cache-dir PATH` | Hugging Face model cache directory. |
| `--alphas A ...` | Intervention strengths to test. Default is `0 1 ... 15`. |
| `--recompute` | Recompute both the baseline and position-probe weights instead of using compatible cached artifacts. |
| `--max-items N` | Debug cap for smoke tests. Omit for full runs. |
| `--aggregate` | Build summary tables from saved `amp_results.pkl` files. |

There is no axis-selection flag. The canonical experiment evaluates horizontal
and vertical spatial relations together.

There is no `--seed` flag. Recomputed probe and random-control values may drift,
while cached artifacts remain loadable.

## Outputs

Results are written to:

```text
coco_amplification/results/<model>/<dtype>/
```

Main output files:

| File | Description |
|------|-------------|
| `amp_results.pkl` | Machine-readable amplification and random-control results. |
| `amp_summary.txt` | Compact sweep summary. |
| `full_report.txt` | Full text report. |

`python -m coco_amplification.plot` writes:

```text
coco_amplification/figures/summary_table.csv
coco_amplification/figures/summary_table.md
```

The bundled Gemma FP32 baseline cache reports strict accuracy `231/440`,
with horizontal `141/279` and vertical `90/161`.

The bundled Pixtral FP32 baseline cache reports strict accuracy
`395/440`, with horizontal `245/279` and vertical `150/161`.

## Layout

```text
coco_amplification/
  run_experiment.py
  plot.py
  setup_repro.sh
  src/
    common.py
    prompts.py
    qwen.py
    gemma.py
    pixtral.py
  results/
  figures/
  scripts/
```
