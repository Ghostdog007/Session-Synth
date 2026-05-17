from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..config.settings import ModelSettings


@dataclass(slots=True)
class HuggingFaceModelAdapter:
    settings: ModelSettings
    processor: Any | None = None
    model: Any | None = None
    tokenizer: Any | None = None

    def token(self) -> str | None:
        return os.getenv(self.settings.token_env_var)

    def load(self) -> None:
        try:
            from transformers import AutoModel, AutoProcessor, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "transformers is not installed. Install the optional hf dependencies before loading a model."
            ) from exc

        model_id = self.settings.resolve_model_id()
        token = self.token()
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.settings.trust_remote_code,
        }
        if token:
            load_kwargs["token"] = token

        if self.settings.device_map is not None:
            load_kwargs["device_map"] = self.settings.device_map
        if self.settings.torch_dtype != "auto":
            load_kwargs["torch_dtype"] = self.settings.torch_dtype
        if self.settings.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        if self.settings.load_in_8bit:
            load_kwargs["load_in_8bit"] = True

        self.processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
        self.model = AutoModel.from_pretrained(model_id, **load_kwargs)

    def ready(self) -> bool:
        return self.model is not None and self.processor is not None

    def describe(self, prompt: str, inputs: dict[str, Any] | None = None) -> str:
        if not self.ready():
            return f"HF placeholder response for {self.settings.resolve_model_id()}: {prompt}"
        return f"Model is loaded, but describe() is still a pipeline placeholder for {self.settings.resolve_model_id()}."
