#!/usr/bin/env python3
"""Run a model-size sweep for MNIST classification with Keras."""

from __future__ import annotations

import argparse
import csv
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # Suppress TensorFlow logging

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from tensorflow import keras


@dataclass
class ExperimentResult:
    hidden_units: int
    trainable_params: int
    test_loss: float
    test_accuracy: float


def parse_hidden_sizes(value: str) -> list[int]:
    sizes = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        parsed = int(token)
        if parsed <= 0:
            raise ValueError("All hidden sizes must be positive integers.")
        sizes.append(parsed)
    if not sizes:
        raise ValueError("Provide at least one hidden size.")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hidden-sizes",
        type=parse_hidden_sizes,
        default=parse_hidden_sizes("8,16,32,64,128,256,512"),
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--competitive-accuracy", type=float, default=0.97)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def build_model(hidden_units: int, learning_rate: float) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28 * 28,)),
            keras.layers.Dense(hidden_units, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_sweep(
    hidden_sizes: Iterable[int],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_split: float,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    for hidden_units in hidden_sizes:
        print(f"\\nTraining model with hidden_units={hidden_units} ...")
        model = build_model(hidden_units=hidden_units, learning_rate=learning_rate)
        model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
        )
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

        result = ExperimentResult(
            hidden_units=hidden_units,
            trainable_params=model.count_params(),
            test_loss=float(test_loss),
            test_accuracy=float(test_accuracy),
        )
        results.append(result)
        print(
            "Result: "
            f"params={result.trainable_params}, "
            f"test_accuracy={result.test_accuracy:.4f}, "
            f"test_loss={result.test_loss:.4f}"
        )

    return sorted(results, key=lambda item: item.trainable_params)


def write_outputs(results: list[ExperimentResult], competitive_accuracy: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mnist_size_study.csv"
    summary_path = output_dir / "mnist_size_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["hidden_units", "trainable_params", "test_loss", "test_accuracy"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))

    competitive_model = next(
        (r for r in results if r.test_accuracy >= competitive_accuracy),
        None,
    )

    summary_payload = {
        "competitive_accuracy_threshold": competitive_accuracy,
        "minimum_competitive_model": asdict(competitive_model) if competitive_model else None,
        "all_results": [asdict(row) for row in results],
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"\\nSaved detailed results to: {csv_path}")
    print(f"Saved summary to: {summary_path}")

    if competitive_model:
        print(
            "Minimum competitive model: "
            f"hidden_units={competitive_model.hidden_units}, "
            f"params={competitive_model.trainable_params}, "
            f"test_accuracy={competitive_model.test_accuracy:.4f}"
        )
    else:
        print(
            "No model met the competitive accuracy threshold. "
            "Try more epochs or larger hidden sizes."
        )


def main() -> None:
    args = parse_args()

    if not (0.0 < args.competitive_accuracy <= 1.0):
        raise ValueError("--competitive-accuracy must be in (0, 1].")
    if not (0.0 < args.validation_split < 1.0):
        raise ValueError("--validation-split must be in (0, 1).")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")

    set_seed(args.seed)
    x_train, y_train, x_test, y_test = load_mnist()

    results = run_sweep(
        hidden_sizes=args.hidden_sizes,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
    )
    write_outputs(results, competitive_accuracy=args.competitive_accuracy, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
