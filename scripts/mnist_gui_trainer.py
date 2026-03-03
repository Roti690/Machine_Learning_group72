#!/usr/bin/env python3
"""Tkinter GUI for training configurable MNIST dense networks."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import queue
import random
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

import numpy as np
import tensorflow as tf
from tensorflow import keras


@dataclass
class TrainResult:
    hidden_layers: list[int]
    trainable_params: int
    test_loss: float
    test_accuracy: float
    elapsed_seconds: float


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


def build_model(hidden_layers: list[int], learning_rate: float) -> keras.Model:
    if not hidden_layers:
        raise ValueError("At least one hidden layer is required.")

    layers: list[keras.layers.Layer] = [keras.layers.Input(shape=(28 * 28,))]
    for units in hidden_layers:
        layers.append(keras.layers.Dense(units, activation="relu"))
    layers.append(keras.layers.Dense(10, activation="softmax"))

    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class EpochLogger(keras.callbacks.Callback):
    def __init__(self, out_queue: queue.Queue[tuple[str, object]]) -> None:
        super().__init__()
        self.out_queue = out_queue

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs or {}
        self.out_queue.put(
            (
                "log",
                (
                    f"Epoch {epoch + 1}: "
                    f"loss={logs.get('loss', 0.0):.4f}, "
                    f"acc={logs.get('accuracy', 0.0):.4f}, "
                    f"val_loss={logs.get('val_loss', 0.0):.4f}, "
                    f"val_acc={logs.get('val_accuracy', 0.0):.4f}"
                ),
            )
        )


class MnistTrainerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MNIST Network Trainer")
        self.root.geometry("760x640")
        self.root.minsize(760, 600)

        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.layer_vars: list[tk.StringVar] = []

        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.x_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

        self.layer_count_var = tk.IntVar(value=2)
        self.epochs_var = tk.StringVar(value="6")
        self.batch_size_var = tk.StringVar(value="128")
        self.learning_rate_var = tk.StringVar(value="0.001")
        self.validation_split_var = tk.StringVar(value="0.1")
        self.seed_var = tk.StringVar(value="42")

        self.status_var = tk.StringVar(value="Ready.")

        self.train_button: ttk.Button | None = None
        self.layers_frame: ttk.Frame | None = None
        self.result_box: tk.Text | None = None
        self.log_box: tk.Text | None = None

        self._build_ui()
        self._rebuild_layer_entries()
        self.root.after(100, self._poll_queue)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(outer, text="Model Setup", padding=10)
        controls.pack(fill="x")

        ttk.Label(controls, text="Hidden layers").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Spinbox(
            controls,
            from_=1,
            to=10,
            textvariable=self.layer_count_var,
            width=8,
            command=self._rebuild_layer_entries,
        ).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(controls, text="Apply", command=self._rebuild_layer_entries).grid(
            row=0, column=2, sticky="w", padx=6, pady=4
        )

        ttk.Label(controls, text="Epochs").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(controls, textvariable=self.epochs_var, width=10).grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(controls, text="Batch size").grid(row=1, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(controls, textvariable=self.batch_size_var, width=10).grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(controls, text="Learning rate").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(controls, textvariable=self.learning_rate_var, width=10).grid(
            row=2, column=1, sticky="w", padx=4, pady=4
        )

        ttk.Label(controls, text="Validation split").grid(row=2, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(controls, textvariable=self.validation_split_var, width=10).grid(
            row=2, column=3, sticky="w", padx=4, pady=4
        )

        ttk.Label(controls, text="Seed").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(controls, textvariable=self.seed_var, width=10).grid(row=3, column=1, sticky="w", padx=4, pady=4)

        self.layers_frame = ttk.LabelFrame(outer, text="Layer Shape", padding=10)
        self.layers_frame.pack(fill="x", pady=10)

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(0, 10))
        self.train_button = ttk.Button(actions, text="Train Network", command=self._start_training)
        self.train_button.pack(side="left")
        ttk.Label(actions, textvariable=self.status_var).pack(side="left", padx=10)

        results = ttk.LabelFrame(outer, text="Results", padding=10)
        results.pack(fill="x", pady=(0, 10))
        results.pack_propagate(False)
        results.configure(height=130)
        self.result_box = tk.Text(results, height=5, wrap="word", state="disabled")
        self.result_box.pack(fill="both", expand=True)
        self._set_result_text("No run yet.")

        logs = ttk.LabelFrame(outer, text="Training Log", padding=10)
        logs.pack(fill="both", expand=True)
        self.log_box = tk.Text(logs, height=18, wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True)

    def _rebuild_layer_entries(self) -> None:
        if self.layers_frame is None:
            return

        for child in self.layers_frame.winfo_children():
            child.destroy()

        layer_count = self.layer_count_var.get()
        if layer_count < 1:
            layer_count = 1
            self.layer_count_var.set(1)

        self.layer_vars = []
        defaults = [128, 64, 32, 16, 8]
        for idx in range(layer_count):
            ttk.Label(self.layers_frame, text=f"Layer {idx + 1} units").grid(
                row=idx,
                column=0,
                sticky="w",
                padx=4,
                pady=3,
            )
            default_units = str(defaults[idx] if idx < len(defaults) else 32)
            var = tk.StringVar(value=default_units)
            self.layer_vars.append(var)
            ttk.Entry(self.layers_frame, textvariable=var, width=12).grid(
                row=idx,
                column=1,
                sticky="w",
                padx=4,
                pady=3,
            )

    def _append_log(self, message: str) -> None:
        if self.log_box is None:
            return
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _set_result_text(self, message: str) -> None:
        if self.result_box is None:
            return
        self.result_box.configure(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.insert("1.0", message)
        self.result_box.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        if self.train_button is None:
            return
        state = "disabled" if busy else "normal"
        self.train_button.configure(state=state)

    def _read_config(self) -> tuple[list[int], int, int, float, float, int]:
        hidden_layers: list[int] = []
        for idx, var in enumerate(self.layer_vars, start=1):
            units = int(var.get())
            if units <= 0:
                raise ValueError(f"Layer {idx} units must be > 0.")
            hidden_layers.append(units)

        epochs = int(self.epochs_var.get())
        batch_size = int(self.batch_size_var.get())
        learning_rate = float(self.learning_rate_var.get())
        validation_split = float(self.validation_split_var.get())
        seed = int(self.seed_var.get())

        if epochs <= 0:
            raise ValueError("Epochs must be > 0.")
        if batch_size <= 0:
            raise ValueError("Batch size must be > 0.")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be > 0.")
        if not (0.0 < validation_split < 1.0):
            raise ValueError("Validation split must be in (0, 1).")

        return hidden_layers, epochs, batch_size, learning_rate, validation_split, seed

    def _start_training(self) -> None:
        try:
            config = self._read_config()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self._append_log("")
        self._append_log("Starting training run...")
        self._set_busy(True)
        self.status_var.set("Training...")
        self._set_result_text("Training in progress...")

        thread = threading.Thread(target=self._train_in_background, args=(config,), daemon=True)
        thread.start()

    def _train_in_background(
        self,
        config: tuple[list[int], int, int, float, float, int],
    ) -> None:
        hidden_layers, epochs, batch_size, learning_rate, validation_split, seed = config
        try:
            set_seed(seed)
            if self.x_train is None:
                self.queue.put(("log", "Loading MNIST dataset..."))
                self.x_train, self.y_train, self.x_test, self.y_test = load_mnist()
                self.queue.put(("log", "MNIST dataset loaded."))

            self.queue.put(("log", f"Model shape: {hidden_layers}"))
            model = build_model(hidden_layers=hidden_layers, learning_rate=learning_rate)
            start = time.perf_counter()
            model.fit(
                self.x_train,
                self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=0,
                callbacks=[EpochLogger(self.queue)],
            )
            test_loss, test_accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
            elapsed = time.perf_counter() - start

            result = TrainResult(
                hidden_layers=hidden_layers,
                trainable_params=model.count_params(),
                test_loss=float(test_loss),
                test_accuracy=float(test_accuracy),
                elapsed_seconds=elapsed,
            )
            self.queue.put(("done", result))
        except Exception as exc:
            self.queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        while True:
            try:
                kind, payload = self.queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._append_log(str(payload))
            elif kind == "done":
                result = payload
                if not isinstance(result, TrainResult):
                    continue
                self.status_var.set("Ready.")
                self._set_busy(False)
                self._set_result_text(
                    "Hidden layers: "
                    f"{result.hidden_layers}\n"
                    f"Trainable params: {result.trainable_params}\n"
                    f"Test accuracy: {result.test_accuracy:.4f}\n"
                    f"Test loss: {result.test_loss:.4f}\n"
                    f"Elapsed: {result.elapsed_seconds:.2f}s"
                )
                self._append_log(
                    "Finished. "
                    f"accuracy={result.test_accuracy:.4f}, "
                    f"loss={result.test_loss:.4f}, "
                    f"params={result.trainable_params}, "
                    f"time={result.elapsed_seconds:.2f}s"
                )
            elif kind == "error":
                self.status_var.set("Failed.")
                self._set_busy(False)
                self._set_result_text("Run failed.")
                messagebox.showerror("Training failed", str(payload))
                self._append_log(f"Error: {payload}")

        self.root.after(100, self._poll_queue)


def main() -> None:
    root = tk.Tk()
    app = MnistTrainerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
