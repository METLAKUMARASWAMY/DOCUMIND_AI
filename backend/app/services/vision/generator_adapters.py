from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.vision.adapters import AdapterStatus, VisionAdapter


class GroundedGeneratorError(RuntimeError):
    pass


class QwenGroundedGeneratorAdapter(VisionAdapter):
    adapter_id = "qwen-grounded-generator"
    stage = "generator"
    model_version = "Qwen3.5-9B"

    def __init__(self) -> None:
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def status(self) -> AdapterStatus:
        installed = all(
            importlib.util.find_spec(package) is not None
            for package in ("transformers", "torch")
        )
        checkpoint = Path(settings.qwen_checkpoint).expanduser()
        assets_available = checkpoint.exists() or settings.qwen_allow_download
        enabled = settings.enable_heavy_models and settings.enable_qwen_local
        active = enabled and installed and assets_available
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else ("disabled" if not enabled else "unavailable"),
            reason=(
                "Qwen grounded generator is ready."
                if active
                else "Enable Qwen and provide a local QWEN_CHECKPOINT or explicitly allow download."
            ),
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
            metadata={
                "checkpoint": str(checkpoint),
                "allow_download": settings.qwen_allow_download,
            },
        )

    def generate(self, question: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        self.ensure_ready()
        import torch

        tokenizer, model = self._load()
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer only from supplied evidence. Return JSON only. Never infer fraud, legal "
                    "judgement, missing amounts, identities, dates, or relationships."
                ),
            },
            {"role": "user", "content": self._prompt(question, evidence)},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(rendered, return_tensors="pt")
        if hasattr(model, "device"):
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][encoded["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return validate_grounded_output(text, evidence)

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            local_only = not settings.qwen_allow_download
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.qwen_checkpoint,
                local_files_only=local_only,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                settings.qwen_checkpoint,
                local_files_only=local_only,
                device_map="auto",
                torch_dtype="auto",
            )
            self._model.eval()
        return self._tokenizer, self._model

    def _prompt(self, question: str, evidence: list[dict[str, Any]]) -> str:
        schema = {
            "answer": "string",
            "answer_status": "supported|partially_supported|not_found",
            "citation_ids": ["evidence-id"],
            "reasoning_summary": "brief evidence summary only",
            "requires_review": False,
        }
        return (
            f"Question: {question}\n"
            f"Evidence: {json.dumps(evidence, ensure_ascii=True)}\n"
            f"Required schema: {json.dumps(schema)}"
        )


class OpenAICompatibleGroundedGenerator(VisionAdapter):
    adapter_id = "openai-compatible-grounded-generator"
    stage = "generator"
    model_version = "configured-provider-model"

    def status(self) -> AdapterStatus:
        installed = importlib.util.find_spec("httpx") is not None
        configured = bool(
            settings.external_llm_base_url
            and settings.external_llm_api_key
            and settings.external_llm_model
        )
        enabled = settings.enable_external_llm
        active = enabled and installed and configured
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else ("disabled" if not enabled else "unavailable"),
            reason=(
                "External grounded generator is configured."
                if active
                else "Configure EXTERNAL_LLM_BASE_URL, API key, and model to enable this provider."
            ),
            enabled=enabled,
            installed=installed,
            model_version=settings.external_llm_model or self.model_version,
        )

    def generate(self, question: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        self.ensure_ready()
        import httpx

        payload = {
            "model": settings.external_llm_model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return JSON with answer, answer_status, citation_ids, reasoning_summary, and "
                        "requires_review. Use only supplied evidence and abstain when evidence is absent."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"question": question, "evidence": evidence}),
                },
            ],
        }
        url = settings.external_llm_base_url.rstrip("/") + "/chat/completions"
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {settings.external_llm_api_key}"},
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        return validate_grounded_output(text, evidence)


class GroundedGeneratorRouter:
    def __init__(self) -> None:
        self.local = QwenGroundedGeneratorAdapter()
        self.external = OpenAICompatibleGroundedGenerator()

    def active(self) -> Any | None:
        if self.external.status().status == "active":
            return self.external
        if self.local.status().status == "active":
            return self.local
        return None


def validate_grounded_output(text: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    payload = _json_object(text)
    answer = str(payload.get("answer", "")).strip()
    status = str(payload.get("answer_status", "not_found"))
    if status not in {"supported", "partially_supported", "not_found"}:
        raise GroundedGeneratorError("Generator returned an unsupported answer_status")
    available_ids = {str(item["id"]) for item in evidence}
    citation_ids = [str(item) for item in payload.get("citation_ids", [])]
    if any(item not in available_ids for item in citation_ids):
        raise GroundedGeneratorError("Generator cited evidence that was not supplied")
    if status != "not_found" and (not answer or not citation_ids):
        raise GroundedGeneratorError("Supported generator answer omitted text or citations")
    evidence_text = " ".join(str(item.get("text", "")) for item in evidence)
    answer_numbers = set(re.findall(r"(?<!\w)-?\d+(?:[.,]\d+)?", answer))
    evidence_numbers = set(re.findall(r"(?<!\w)-?\d+(?:[.,]\d+)?", evidence_text))
    if not answer_numbers.issubset(evidence_numbers):
        raise GroundedGeneratorError("Generator introduced a number absent from evidence")
    return {
        "answer": answer,
        "answer_status": status,
        "citation_ids": citation_ids,
        "reasoning_summary": str(payload.get("reasoning_summary", ""))[:320],
        "requires_review": bool(payload.get("requires_review", False)),
    }


def _json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise GroundedGeneratorError("Generator did not return JSON")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise GroundedGeneratorError("Generator returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise GroundedGeneratorError("Generator JSON root must be an object")
    return value
