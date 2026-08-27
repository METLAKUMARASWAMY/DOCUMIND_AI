"""Model registry administration, rule overrides and model comparison.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.core.security import utc_now_iso
from app.models.vision import (
    ReviewerStatus,
)
from app.services.vision.deterministic_extraction import deterministic_schema_extractor
from app.services.vision.schema_registry import vision_schema_registry


class AdminMixin:
    def model_comparison(
        self,
        document_id: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        document = self.get_document(document_id, tenant_id)
        registry = self.model_registry.registry()
        schema = vision_schema_registry.get(document.document_type)
        selected = {field.name: field for field in document.extracted_fields}
        deterministic = {field.name: field for field in self._extract_fields(document)}
        donut_payload = dict(document.model_routing.get("donut_comparison") or {})
        donut_values = self._flatten_model_output(donut_payload.get("output") or {})
        field_names = list(
            dict.fromkeys(
                [
                    *([field.name for field in schema.fields] if schema else []),
                    *selected,
                    *deterministic,
                    *(key.rsplit(".", 1)[-1] for key in donut_values),
                ]
            )
        )
        extraction_runs = [run for run in document.model_runs if run.stage == "extraction"]
        primary_run = next(
            (run for run in reversed(extraction_runs) if run.status == "completed"),
            extraction_runs[-1] if extraction_runs else None,
        )
        availability = [
            item
            for item in registry["backends"]
            if item["stage"]
            in {
                "classification_extraction",
                "ocr_free_baseline",
                "advanced_parsing",
            }
        ]

        rows = []
        for field_name in field_names:
            selected_field = selected.get(field_name)
            fallback_field = deterministic.get(field_name)
            layout_field = (
                selected_field
                if selected_field and "layoutlm" in selected_field.model_name.lower()
                else None
            )
            donut_value = next(
                (
                    value
                    for key, value in donut_values.items()
                    if key.rsplit(".", 1)[-1] == field_name.lower()
                ),
                None,
            )
            observed_values = [
                value
                for value in (
                    selected_field.value if selected_field else None,
                    fallback_field.value if fallback_field else None,
                    str(donut_value) if donut_value is not None else None,
                )
                if value is not None
            ]
            disagreement = len({self._normalize(value) for value in observed_values}) > 1
            required = bool(
                schema and schema.field(field_name) and schema.field(field_name).required
            )
            if selected_field:
                validation_result = (
                    "accepted"
                    if selected_field.reviewer_status
                    in {ReviewerStatus.accepted, ReviewerStatus.edited}
                    else "requires_review"
                    if selected_field.confidence is None
                    or selected_field.confidence < self._mandatory_review_threshold(document)
                    else "unverified"
                )
            else:
                validation_result = "missing_required" if required else "not_extracted"
            rows.append(
                {
                    "field_id": selected_field.field_id if selected_field else None,
                    "field_name": field_name,
                    "required": required,
                    "primary_result": self._mask_sensitive(field_name, selected_field.value)
                    if selected_field
                    else None,
                    "primary_confidence": selected_field.confidence if selected_field else None,
                    "primary_model": selected_field.model_name if selected_field else None,
                    "fallback_result": self._mask_sensitive(field_name, fallback_field.value)
                    if fallback_field
                    else None,
                    "fallback_confidence": fallback_field.confidence if fallback_field else None,
                    "fallback_model": fallback_field.model_name if fallback_field else None,
                    "layoutlmv3_result": self._mask_sensitive(field_name, layout_field.value)
                    if layout_field
                    else None,
                    "layoutlmv3_confidence": layout_field.confidence if layout_field else None,
                    "donut_result": self._mask_sensitive(field_name, str(donut_value))
                    if donut_value is not None
                    else None,
                    "donut_confidence": None,
                    "validation_result": validation_result,
                    "selected_result": self._mask_sensitive(
                        field_name, selected_field.reviewed_value or selected_field.value
                    )
                    if selected_field
                    else None,
                    "review_status": selected_field.reviewer_status.value
                    if selected_field
                    else "unresolved"
                    if required
                    else "not_available",
                    "disagreement": disagreement,
                    "evidence": selected_field.evidence.model_dump(mode="json")
                    if selected_field
                    else None,
                }
            )
        return {
            "document_id": document_id,
            "comparison_mode": "observed_runtime_outputs",
            "primary_adapter": {
                "model_id": primary_run.model_name if primary_run else None,
                "status": primary_run.status if primary_run else "not_recorded",
                "runtime_ms": primary_run.processing_duration_ms if primary_run else None,
                "device": primary_run.device if primary_run else None,
                "checkpoint": primary_run.checkpoint if primary_run else None,
            },
            "fallback_adapter": {
                "model_id": deterministic_schema_extractor.adapter_id,
                "status": "observed" if deterministic else "no_fields_observed",
                "field_count": len(deterministic),
            },
            "availability": availability,
            "field_output": rows,
            "fallback_routing": document.model_routing,
            "routing_reason": document.model_routing.get("reasons", []),
            "model_runs": [run.model_dump(mode="json") for run in document.model_runs],
            "note": (
                "Only observed outputs are shown. Unavailable model cells stay blank until a "
                "checkpoint-backed adapter completes inference for this document."
            ),
        }

    def _flatten_model_output(self, payload: Any) -> dict[str, Any]:
        flattened: dict[str, Any] = {}

        def visit(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(f"{prefix}.{key}" if prefix else str(key), child)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(f"{prefix}.{index}", child)
            elif value is not None:
                flattened[prefix.lower().replace(" ", "_")] = value

        visit("", payload)
        return flattened

    def admin_models(self) -> dict[str, Any]:
        registry = self.model_registry.registry()
        for backend in registry["backends"]:
            override = self.model_overrides.get(str(backend["model_id"]))
            backend["requested"] = (
                bool(override["requested_enabled"])
                if override and override.get("requested_enabled") is not None
                else bool(backend.get("enabled"))
            )
        return {
            "registry": registry,
            "overrides": self.model_overrides,
            "smoke_results": self.model_smoke.store.all(),
            "policy": (
                "Requested state is deployment intent only. Active status requires the current "
                "package, checkpoint, adapter initialization, inference, and output-schema smoke test."
            ),
        }

    def admin_model_health(self, model_id: str) -> dict[str, Any]:
        models = self.admin_models()["registry"]["backends"]
        match = next(
            (
                item
                for item in models
                if str(item["model_id"]) == model_id or str(item.get("adapter_id")) == model_id
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Model is not registered: {model_id}")
        return match

    def patch_admin_model(
        self,
        model_id: str,
        payload: dict[str, Any],
        *,
        actor: str = "system",
        tenant_id: str = "tenant-local",
    ) -> dict[str, Any]:
        registered = {
            str(item["model_id"])
            for item in self.model_registry.backends(include_runtime_health=False)
        }
        if model_id not in registered:
            raise ValueError(f"Model is not registered: {model_id}")
        override = {
            "model_id": model_id,
            "requested_enabled": payload.get("requested", payload.get("enabled")),
            "profile": payload.get("profile"),
            "notes": payload.get("notes", ""),
            "actor": actor,
            "updated_at": utc_now_iso(),
            "status": "recorded_for_deployment_review",
        }
        self.model_overrides[model_id] = override
        self._write_json(self._state_root / "model_overrides.json", self.model_overrides)
        self._system_audit(
            "model_state_change",
            "Recorded a requested model deployment state without changing runtime availability.",
            actor=actor,
            tenant_id=tenant_id,
            metadata={
                "model_id": model_id,
                "requested_enabled": override["requested_enabled"],
                "status": override["status"],
            },
        )
        return override

    def smoke_test_admin_model(
        self,
        model_id: str,
        *,
        actor: str = "system",
        tenant_id: str = "tenant-local",
    ) -> dict[str, Any]:
        result = self.model_smoke.run(model_id)
        self._system_audit(
            "model_smoke_test",
            "Executed adapter initialization, inference, and output-schema verification.",
            actor=actor,
            tenant_id=tenant_id,
            metadata={
                "model_id": result["model_id"],
                "status": result["status"],
                "health_status": result["health_status"],
                "configuration_fingerprint": result["configuration_fingerprint"],
            },
        )
        return result

    def admin_rules(self) -> dict[str, Any]:
        return {
            "confidence_policy": self.confidence_policy,
            "validation_rules": [
                "amount_arithmetic",
                "due_date_after_invoice_date",
                "approved_bank_match",
                "required_signature",
                "vendor_tax_id_format",
                "payment_terms_conflict",
                "duplicate_detection",
                "advanced_fallback_review",
            ],
            "overrides": self.rule_overrides,
        }

    def patch_admin_rule(
        self,
        rule_id: str,
        payload: dict[str, Any],
        *,
        actor: str = "system",
        tenant_id: str = "tenant-local",
    ) -> dict[str, Any]:
        override = {
            "rule_id": rule_id,
            "enabled": payload.get("enabled"),
            "threshold": payload.get("threshold"),
            "notes": payload.get("notes", ""),
            "actor": actor,
            "updated_at": utc_now_iso(),
            "status": "recorded_for_local_runtime",
        }
        self.rule_overrides[rule_id] = override
        self._write_json(self._state_root / "rule_overrides.json", self.rule_overrides)
        self._system_audit(
            "rule_state_change",
            "Recorded a validation-rule override.",
            actor=actor,
            tenant_id=tenant_id,
            metadata={
                "rule_id": rule_id,
                "enabled": override["enabled"],
                "threshold": override["threshold"],
            },
        )
        return override

    def _load_admin_overrides(self) -> None:
        payload = self._read_json(self._state_root / "model_overrides.json") or {}
        self.model_overrides.update(
            {
                str(model_id): dict(value)
                for model_id, value in payload.items()
                if isinstance(value, dict)
            }
        )
        rule_payload = self._read_json(self._state_root / "rule_overrides.json") or {}
        self.rule_overrides.update(
            {
                str(rule_id): dict(value)
                for rule_id, value in rule_payload.items()
                if isinstance(value, dict)
            }
        )
        audit_payload = self._read_json(self._state_root / "system_audit.json") or {}
        self.system_audit_events.extend(
            dict(value) for value in audit_payload.get("events", []) if isinstance(value, dict)
        )

    def _system_audit(
        self,
        event_type: str,
        message: str,
        *,
        actor: str,
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "event_id": f"sys-aud-{uuid4().hex[:12]}",
            "document_id": "system",
            "tenant_id": tenant_id,
            "actor": actor,
            "event_type": event_type,
            "message": message,
            "created_at": utc_now_iso(),
            "metadata": metadata or {},
        }
        self.system_audit_events.append(event)
        self._write_json(
            self._state_root / "system_audit.json",
            {"events": self.system_audit_events},
        )
        return event
