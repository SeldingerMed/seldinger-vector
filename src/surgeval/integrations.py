"""Integrations for PyTorch, Hugging Face, Gymnasium, and RLlib models."""

from __future__ import annotations

from typing import Any


class PyTorchPolicyWrapper:
    """Wrapper adapting a PyTorch neural network into a SurgEval policy."""

    def __init__(self, model: Any, *, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    def reset(self, *, seed: int | None = None) -> None:
        """Reset internal recurrent/RNN states if present."""
        reset_fn = getattr(self.model, "reset", None)
        if callable(reset_fn):
            reset_fn(seed=seed)

    def act(self, observation: Any, *, step: int = 0) -> Any:
        """Compute action from observation via model forward pass."""
        try:
            import numpy as np
            import torch

            with torch.no_grad():
                if isinstance(observation, dict):
                    # Tensorize numeric arrays in dict
                    tensor_obs = {
                        k: torch.as_tensor(v).unsqueeze(0)
                        if isinstance(v, (list, tuple, np.ndarray))
                        else v
                        for k, v in observation.items()
                    }
                    out = self.model(tensor_obs)
                elif isinstance(observation, (list, tuple, np.ndarray)):
                    tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                    out = self.model(tensor)
                else:
                    out = self.model(observation)

                if hasattr(out, "detach"):
                    arr = out.detach().cpu().numpy()
                    if getattr(arr, "ndim", 0) > 1 and arr.shape[0] == 1:
                        return arr.squeeze(0)
                    return arr
                return out
        except ImportError:
            # Fallback if torch is not installed in local environment
            act_fn = getattr(self.model, "act", getattr(self.model, "forward", self.model))
            return act_fn(observation)


class HuggingFacePredictorWrapper:
    """Wrapper adapting a Hugging Face pipeline or VLM into a SurgEval predictor."""

    def __init__(self, pipeline_or_model: Any) -> None:
        self.pipeline = pipeline_or_model

    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        """Generate structured prediction from input item."""
        if callable(self.pipeline):
            result = self.pipeline(item)
            if isinstance(result, dict):
                return result
            if (
                isinstance(result, (list, tuple))
                and len(result) > 0
                and isinstance(result[0], dict)
            ):
                return result[0]
            return {"prediction": result}
        return {"prediction": str(self.pipeline)}


class GymnasiumPolicyWrapper:
    """Wrapper adapting standard RL policies (Stable-Baselines3, Ray RLlib, CleanRL)."""

    def __init__(self, policy: Any, *, deterministic: bool = True) -> None:
        self.policy = policy
        self.deterministic = deterministic

    def reset(self, *, seed: int | None = None) -> None:
        reset_fn = getattr(self.policy, "reset", None)
        if callable(reset_fn):
            reset_fn(seed=seed)

    def act(self, observation: Any, *, step: int = 0) -> Any:
        predict_fn = getattr(self.policy, "predict", None)
        if callable(predict_fn):
            try:
                res = predict_fn(observation, deterministic=self.deterministic)
            except TypeError:
                res = predict_fn(observation)
            return res[0] if isinstance(res, tuple) else res
        act_fn = getattr(self.policy, "act", self.policy)
        return act_fn(observation)


def wrap_pytorch(model: Any, *, deterministic: bool = True) -> PyTorchPolicyWrapper:
    """Wrap a PyTorch model for SurgEval evaluation."""
    return PyTorchPolicyWrapper(model, deterministic=deterministic)


def wrap_hf(pipeline_or_model: Any) -> HuggingFacePredictorWrapper:
    """Wrap a Hugging Face pipeline/model for SurgEval evaluation."""
    return HuggingFacePredictorWrapper(pipeline_or_model)


def wrap_gym_policy(policy: Any, *, deterministic: bool = True) -> GymnasiumPolicyWrapper:
    """Wrap a Gymnasium/RLlib policy for SurgEval evaluation."""
    return GymnasiumPolicyWrapper(policy, deterministic=deterministic)
