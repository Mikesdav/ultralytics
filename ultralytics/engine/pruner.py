# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from torch.nn.utils import prune

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, LOGGER, YAML, callbacks


class Pruner:
    """A class for pruning YOLO models with a unified interface."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.args = get_cfg(cfg, overrides)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def __call__(self, model=None) -> str:
        if model is None:
            raise ValueError("Pruner requires a PyTorch model instance.")
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Pruning only supports PyTorch models.")

        save_dir = get_save_dir(self.args)
        save_dir.mkdir(parents=True, exist_ok=True)
        weights_path = save_dir / "pruned.pt"

        self.run_callbacks("on_prune_start")
        pruned_model = deepcopy(model).cpu().eval()
        parameters = self._parameters_to_prune(pruned_model)
        if not parameters:
            raise ValueError("No pruneable parameters found in model.")

        method = self.args.prune_method.lower()
        amount = float(self.args.prune_amount)
        if method == "l1_unstructured":
            self._apply_unstructured(pruned_model, parameters, amount, prune.L1Unstructured)
        elif method == "random_unstructured":
            self._apply_unstructured(pruned_model, parameters, amount, prune.RandomUnstructured)
        elif method == "lamp_unstructured":
            if not self.args.prune_global:
                raise ValueError("lamp_unstructured requires prune_global=True.")
            self._apply_lamp_unstructured(parameters, amount)
        elif method == "nvidia_modelopt":
            self._apply_modelopt_pruning(pruned_model)
        elif method == "ln_structured":
            self._apply_ln_structured(pruned_model, parameters, amount)
        else:
            raise ValueError(
                f"Invalid prune_method='{self.args.prune_method}'. "
                "Valid options are l1_unstructured, random_unstructured, lamp_unstructured, ln_structured, nvidia_modelopt."
            )

        if self.args.prune_remove:
            self._remove_reparametrization(pruned_model)

        self._save_checkpoint(pruned_model, weights_path)
        self.run_callbacks("on_prune_end")
        LOGGER.info(f"Pruned model saved to {weights_path}")
        return str(weights_path)

    @staticmethod
    def _parameters_to_prune(model: torch.nn.Module) -> list[tuple[torch.nn.Module, str]]:
        parameters = []
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                parameters.append((module, "weight"))
        return parameters

    def _apply_unstructured(
        self,
        model: torch.nn.Module,
        parameters: list[tuple[torch.nn.Module, str]],
        amount: float,
        method,
    ) -> None:
        if self.args.prune_global:
            prune.global_unstructured(parameters, pruning_method=method, amount=amount)
            return
        for module, name in parameters:
            method.apply(module, name, amount)

    def _apply_ln_structured(
        self,
        model: torch.nn.Module,
        parameters: list[tuple[torch.nn.Module, str]],
        amount: float,
    ) -> None:
        if self.args.prune_global:
            raise ValueError("Global pruning is not supported for ln_structured.")
        for module, name in parameters:
            prune.ln_structured(module, name, amount=amount, n=self.args.prune_n, dim=self.args.prune_dim)

    @staticmethod
    def _apply_lamp_unstructured(parameters: list[tuple[torch.nn.Module, str]], amount: float) -> None:
        if amount <= 0.0:
            return
        with torch.no_grad():
            scores = []
            shapes = []
            devices = []
            for module, name in parameters:
                weight = getattr(module, name).detach()
                abs_w = weight.abs().flatten()
                if abs_w.numel() == 0:
                    scores.append(abs_w)
                    shapes.append(weight.shape)
                    devices.append(weight.device)
                    continue
                sorted_vals, idx = torch.sort(abs_w)
                cumsum = torch.cumsum(sorted_vals, 0)
                total = cumsum[-1]
                denom = total - torch.cat([torch.zeros(1, device=abs_w.device), cumsum[:-1]])
                normalized_sorted = sorted_vals / torch.clamp(denom, min=1e-12)
                normalized = torch.empty_like(abs_w)
                normalized[idx] = normalized_sorted
                scores.append(normalized)
                shapes.append(weight.shape)
                devices.append(weight.device)
            flat_scores = torch.cat(scores) if scores else torch.tensor([])
            total_params = flat_scores.numel()
            if total_params == 0:
                return
            k = int(total_params * amount)
            if k >= total_params:
                threshold = flat_scores.max()
            else:
                threshold = torch.kthvalue(flat_scores, k).values if k > 0 else flat_scores.min() - 1
            for (module, name), shape, device, score in zip(parameters, shapes, devices, scores):
                numel = score.numel()
                if numel == 0:
                    continue
                mask = (score > threshold).to(device=device).view(shape)
                prune.custom_from_mask(module, name, mask)

    def _apply_modelopt_pruning(self, model: torch.nn.Module) -> None:
        try:
            import importlib

            pruning = importlib.import_module("modelopt.torch.pruning")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "NVIDIA ModelOpt is not installed. Install with `pip install nvidia-modelopt`."
            ) from exc

        config = self._resolve_modelopt_config()
        if config is None:
            raise ValueError(
                "prune_modelopt_config is required for nvidia_modelopt. Provide a dict or path to YAML/JSON."
            )
        prune_model = getattr(pruning, "prune_model", None)
        if prune_model is None:
            raise AttributeError("modelopt.torch.pruning.prune_model is required for nvidia_modelopt.")
        try:
            prune_model(model, config)
        except TypeError:
            if isinstance(config, dict):
                prune_model(model, **config)
            else:
                raise

    def _resolve_modelopt_config(self) -> dict | None:
        config = self.args.prune_modelopt_config
        if config is None:
            return None
        if isinstance(config, dict):
            return config
        if isinstance(config, str) and config:
            path = Path(config)
            if path.suffix.lower() in {".yaml", ".yml", ".json"} and path.exists():
                return YAML.load(path)
        raise ValueError("prune_modelopt_config must be a dict or path to an existing YAML/JSON file.")

    @staticmethod
    def _remove_reparametrization(model: torch.nn.Module) -> None:
        for module in model.modules():
            if hasattr(module, "weight_orig"):
                prune.remove(module, "weight")

    def _save_checkpoint(self, model: torch.nn.Module, weights_path: Path) -> None:
        updates = {
            "model": model.half(),
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "train_args": getattr(model, "args", {}),
        }
        torch.save(updates, weights_path)

    def add_callback(self, event: str, callback):
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)
