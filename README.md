# MNIST Minimum Network Size Study (Keras)

Research question:

> What is the minimum network size that achieves competitive accuracy in MNIST classification?

This project runs a controlled sweep over simple feed-forward networks (single hidden layer) and reports the smallest model that reaches a target test accuracy.

## Project layout

- `scripts/run_mnist_size_study.py`: main experiment script
- `results/`: generated CSV + JSON summaries
- `requirements.txt`: Python dependencies
- `Makefile`: convenience commands for venv + run

## Quick start

```bash
cd /home/david/Projects/Machine_learning
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_mnist_size_study.py
```

Or with Make:

```bash
make install
make run
```

## Example custom run

```bash
python scripts/run_mnist_size_study.py \
  --hidden-sizes 8,16,32,64,128,256,512 \
  --epochs 8 \
  --batch-size 128 \
  --competitive-accuracy 0.97 \
  --seed 42
```

## Outputs

After each run, the script writes:

- `results/mnist_size_study.csv`
- `results/mnist_size_summary.json`

The summary JSON contains the smallest model that meets the `--competitive-accuracy` target.

## Notes

- Network size is measured by trainable parameter count.
- The baseline architecture is intentionally simple to isolate capacity effects.
- You can tune learning rate, epochs, and hidden sizes from CLI flags.
