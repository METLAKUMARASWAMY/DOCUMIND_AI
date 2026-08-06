from __future__ import annotations

import importlib.util
import json
import re
from typing import Any

from app.core.config import settings
from app.services.vision.adapters import AdapterStatus, VisionAdapter
from app.services.vision.checkpoint_utils import local_or_hf_cached_checkpoint


class GroundedGeneratorError(RuntimeError):
    pass


class GroundedTemplateGeneratorAdapter(VisionAdapter):
    adapter_id = "grounded-template-generator"
    stage = "generator"
    model_version = "grounded-template-generator-v1"

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active",
            reason="Deterministic evidence-only response composition is available.",
            enabled=True,
            installed=True,
            model_version=self.model_version,
            metadata={"generative_model": False, "evidence_required": True},
        )

    def generate(self, question: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        del question
        usable = [
            item
            for item in evidence
            if str(item.get("id", "")).strip() and str(item.get("text", "")).strip()
        ]
        if not usable:
            return {
                "answer": "The available documents do not contain sufficient evidence to answer this question.",
                "answer_status": "not_found",
                "citation_ids": [],
                "reasoning_summary": "No source evidence was available.",
                "requires_review": False,
            }
        selected = usable[:3]
        return {
            "answer": " ".join(str(item["text"]).strip() for item in selected)[:1200],
            "answer_status": "supported",
            "citation_ids": [str(item["id"]) for item in selected],
            "reasoning_summary": "Composed directly from the selected evidence blocks.",
            "requires_review": any(
                str(item.get("review_status", "unverified"))
                not in {"accepted", "edited"}
                for item in selected
            ),
        }


class QwenGroundedGeneratorAdapter(VisionAdapter):
    """Local Transformers instruction generator (legacy adapter import name)."""

    adapter_id = "qwen-grounded-generator"
    stage = "generator"
    model_version = "configured-local-transformers-model"

    def __init__(self) -> None:
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def status(self) -> AdapterStatus:
        installed = all(
            importlib.util.find_spec(package) is not None
            for package in ("transformers", "torch")
        )
        checkpoint = settings.qwen_checkpoint
        local_assets = local_or_hf_cached_checkpoint(checkpoint)
        assets_available = local_assets is not None or settings.qwen_allow_download
        enabled = settings.enable_heavy_models and settings.enable_qwen_local
        active = enabled and installed and assets_available
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else ("disabled" if not enabled else "unavailable"),
            reason=(
                f"Local Transformers generator {checkpoint} is configured."
                if active
                else "Enable the local generator and provide cached/local assets or explicitly allow download."
            ),
            enabled=enabled,
            installed=installed,
            model_version=checkpoint,
            metadata={
                "checkpoint": checkpoint,
                "resolved_checkpoint": str(local_assets) if local_assets else None,
                "allow_download": settings.qwen_allow_download,
                "structured_output_guard": "strict-json-or-grounded-text-normalizer-v1",
            },
        )

    def generate(self, question: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        self.ensure_ready()

        if not evidence:
            return {
                "answer": "The available documents do not contain sufficient evidence to answer this question.",
                "answer_status": "not_found",
                "citation_ids": [],
                "reasoning_summary": "No source evidence was supplied to the generator.",
                "requires_review": False,
            }
        tokenizer, model = self._load()
        rendered = self._render_prompt(tokenizer, question, evidence)
        text = self._generate_text(tokenizer, model, rendered, max_new_tokens=256)
        try:
            return validate_grounded_output(text, evidence)
        except GroundedGeneratorError:
            # Small local instruction models often answer faithfully but cannot emit the
            # requested JSON schema. A second real inference asks for plain text; the
            # deterministic normalizer accepts it only when its claims are covered by
            # supplied evidence and then attaches only source IDs from that evidence.
            plain_prompt = self._plain_grounded_prompt(question, evidence)
            plain_text = self._generate_text(tokenizer, model, plain_prompt, max_new_tokens=96)
            return normalize_grounded_text(plain_text, evidence)

    def _generate_text(
        self,
        tokenizer: Any,
        model: Any,
        rendered: str,
        *,
        max_new_tokens: int,
    ) -> str:
        import torch

        encoded = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=2048)
        if hasattr(model, "device"):
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = (
            output[0]
            if bool(getattr(model.config, "is_encoder_decoder", False))
            else output[0][encoded["input_ids"].shape[1] :]
        )
        return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )

            local_only = not settings.qwen_allow_download
            config = AutoConfig.from_pretrained(
                settings.qwen_checkpoint,
                local_files_only=local_only,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.qwen_checkpoint,
                local_files_only=local_only,
            )
            model_factory = (
                AutoModelForSeq2SeqLM if bool(config.is_encoder_decoder) else AutoModelForCausalLM
            )
            self._model = model_factory.from_pretrained(
                settings.qwen_checkpoint,
                local_files_only=local_only,
            )
            self._model.eval()
        return self._tokenizer, self._model

    def _render_prompt(
        self,
        tokenizer: Any,
        question: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        task = self._prompt(question, evidence)
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            return tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "Answer only from supplied evidence. Return JSON only. Never infer "
                            "missing facts, identities, dates, numbers, or relationships."
                        ),
                    },
                    {"role": "user", "content": task},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            "Answer the task using only the supplied evidence. Output one JSON object and no "
            f"other text.\n{task}"
        )

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

    def _plain_grounded_prompt(
        self,
        question: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        evidence_text = " ".join(str(item.get("text", "")).strip() for item in evidence)
        return (
            f"Question: {question}\n"
            f"Evidence: {evidence_text}\n"
            "Answer using only the evidence in one short sentence:"
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


def normalize_grounded_text(text: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap plain model text only after deterministic evidence-coverage checks."""

    answer = text.strip().strip('"').strip()
    if not answer or len(answer) > 1200 or any(marker in answer for marker in ("{", "[")):
        raise GroundedGeneratorError("Generator plain-text fallback was empty or malformed")

    answer_tokens = _grounding_tokens(answer)
    answer_numbers = set(re.findall(r"(?<!\w)-?\d+(?:[.,]\d+)?", answer))
    ranked: list[tuple[float, dict[str, Any]]] = []
    for item in evidence:
        item_id = str(item.get("id", "")).strip()
        item_text = str(item.get("text", "")).strip()
        if not item_id or not item_text:
            continue
        evidence_tokens = _grounding_tokens(item_text)
        evidence_numbers = set(re.findall(r"(?<!\w)-?\d+(?:[.,]\d+)?", item_text))
        if answer_numbers and not answer_numbers.issubset(evidence_numbers):
            continue
        overlap = len(answer_tokens & evidence_tokens)
        coverage = overlap / max(1, len(answer_tokens))
        if overlap or (answer_numbers and answer_numbers.issubset(evidence_numbers)):
            ranked.append((coverage, item))

    if not ranked:
        raise GroundedGeneratorError("Generator plain-text fallback was not covered by evidence")
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    coverage, selected = ranked[0]
    if answer_tokens and coverage < 0.5:
        raise GroundedGeneratorError("Generator plain-text fallback had insufficient evidence coverage")

    selected_id = str(selected["id"])
    payload = {
        "answer": answer,
        "answer_status": "supported",
        "citation_ids": [selected_id],
        "reasoning_summary": "The generated answer was checked against the cited evidence.",
        "requires_review": str(selected.get("review_status", "unverified"))
        not in {"accepted", "edited"},
    }
    return validate_grounded_output(json.dumps(payload), evidence)


def _grounding_tokens(value: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "is",
        "of",
        "the",
        "to",
        "was",
    }
    return {
        token
        for token in re.findall(r"[\w.-]+", value.lower())
        if token not in stopwords
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
