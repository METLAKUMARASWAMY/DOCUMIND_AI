from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from statistics import mean
from typing import Any

from app.core.security import utc_now_iso

NO_ANSWER_TEXT = (
    "the available documents do not contain sufficient evidence to answer this question"
)


class VisionEvaluationService:
    """Calculates metrics only from caller-supplied references and judgements."""

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        references = payload.get("references") or []
        predictions = payload.get("predictions") or []
        if not references or not predictions:
            return self._not_calculated(
                "Labelled references and predictions are required before metrics can be reported."
            )
        if not isinstance(references, list) or not isinstance(predictions, list):
            return self._not_calculated("references and predictions must be JSON arrays.")
        if len(references) != len(predictions):
            return self._not_calculated(
                "references and predictions must contain the same number of samples."
            )

        dataset = self._dataset_provenance(payload.get("dataset"))
        metrics: dict[str, Any] = {}
        if self._has_any(references, "ocr_text") and self._has_any(predictions, "ocr_text"):
            metrics["ocr"] = self.ocr_metrics(references, predictions)
        if self._has_any(references, "document_type"):
            metrics["classification"] = self.classification_metrics(
                [str(item.get("document_type", "")) for item in references],
                [str(item.get("document_type", "")) for item in predictions],
            )
        if self._has_any(references, "fields"):
            metrics["extraction"] = self.field_metrics(
                [self._mapping(item.get("fields")) for item in references],
                [self._mapping(item.get("fields")) for item in predictions],
            )
        if self._has_detection_data(references):
            metrics["layout_and_detection"] = self.detection_metrics(
                references,
                predictions,
            )
        if self._has_any(references, "tables"):
            metrics["tables"] = self.table_metrics(references, predictions)
        if self._has_any(references, "answer") or self._has_any(references, "answers"):
            metrics["qa"] = self.qa_metrics(
                [item.get("answers", item.get("answer", "")) for item in references],
                [item.get("answers", item.get("answer", "")) for item in predictions],
            )

        if payload.get("retrieval"):
            metrics["retrieval"] = self.retrieval_metrics(payload["retrieval"])
        if payload.get("rag_judgements"):
            metrics["rag"] = self.rag_metrics(payload["rag_judgements"])
        if payload.get("risk"):
            metrics["risk"] = self.risk_metrics(
                payload["risk"],
                float(payload.get("review_capacity", 0.10)),
            )
        if payload.get("confidence"):
            metrics["confidence"] = self.confidence_metrics(
                payload["confidence"],
                int(payload.get("calibration_bins", 10)),
            )
        if payload.get("human_review"):
            metrics["human_review"] = self.human_review_metrics(payload["human_review"])
        if payload.get("system_measurements"):
            metrics["system"] = self.system_metrics(payload["system_measurements"])

        result: dict[str, Any] = {
            "status": "calculated",
            "sample_count": len(references),
            "dataset": dataset,
            "report_eligible": bool(
                dataset.get("name")
                and dataset.get("version")
                and dataset.get("split") in {"validation", "test"}
                and dataset.get("provenance")
            ),
            "calculated_at": utc_now_iso(),
            "input_digest": self._digest(payload),
            "metrics": metrics,
            "fake_metrics_reported": False,
            "metric_policy": (
                "All values were calculated from the supplied labels, scores, judgements, or "
                "measurements; no missing metric was estimated."
            ),
        }
        # Preserve the established API shape while exposing the grouped metrics object.
        result.update(metrics)
        return result

    def supported_metrics(self) -> dict[str, list[str]]:
        return {
            "ocr": [
                "character_error_rate",
                "word_error_rate",
                "word_detection_precision",
                "word_detection_recall",
            ],
            "classification": [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "confusion_matrix",
            ],
            "extraction": [
                "field_precision",
                "field_recall",
                "field_f1",
                "exact_match",
                "normalized_value_accuracy",
            ],
            "layout_and_detection": [
                "mean_iou",
                "map50",
                "map50_95",
                "per_class_precision",
                "per_class_recall",
            ],
            "tables": [
                "cell_precision",
                "cell_recall",
                "cell_f1",
                "structure_accuracy",
                "teds",
            ],
            "qa": ["exact_match", "anls", "list_answer_anls", "no_answer_accuracy"],
            "retrieval": [
                "recall_at_k",
                "precision_at_k",
                "mrr",
                "ndcg_at_k",
                "evidence_document_map",
            ],
            "rag": [
                "faithfulness",
                "answer_relevance",
                "context_relevance",
                "citation_correctness",
                "unsupported_claim_rate",
                "hallucination_rate",
            ],
            "risk": [
                "precision",
                "recall",
                "f1",
                "auprc",
                "false_positive_rate",
                "recall_at_fixed_review_budget",
            ],
            "confidence": ["expected_calibration_error", "aurc", "risk_coverage_curve"],
            "human_review": [
                "acceptance_rate",
                "correction_rate",
                "mean_review_time_seconds",
                "escalation_rate",
                "reviewer_agreement",
                "critical_error_escape_rate",
            ],
            "system": [
                "latency_per_page",
                "full_document_latency",
                "throughput",
                "peak_memory",
                "queue_time",
                "retrieval_latency",
                "failure_rate",
            ],
        }

    def ocr_metrics(
        self,
        references: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> dict[str, float]:
        character_rates = []
        word_rates = []
        true_positive = false_positive = false_negative = 0
        for reference, prediction in zip(references, predictions, strict=True):
            reference_text = str(reference.get("ocr_text", ""))
            prediction_text = str(prediction.get("ocr_text", ""))
            character_rates.append(self.character_error_rate(reference_text, prediction_text))
            word_rates.append(self.word_error_rate(reference_text, prediction_text))
            reference_words = self._records(reference.get("ocr_words"))
            predicted_words = sorted(
                self._records(prediction.get("ocr_words")),
                key=lambda item: float(item.get("confidence", 0.0)),
                reverse=True,
            )
            matched: set[int] = set()
            for predicted in predicted_words:
                best = self._best_box_match(predicted, reference_words, matched, require_label=False)
                if best is not None and self._iou(predicted, reference_words[best]) >= 0.5:
                    matched.add(best)
                    true_positive += 1
                else:
                    false_positive += 1
            false_negative += len(reference_words) - len(matched)
        return {
            "character_error_rate": self._mean(character_rates),
            "word_error_rate": self._mean(word_rates),
            "word_detection_precision": self._ratio(
                true_positive,
                true_positive + false_positive,
            ),
            "word_detection_recall": self._ratio(
                true_positive,
                true_positive + false_negative,
            ),
        }

    def classification_metrics(
        self,
        references: list[str],
        predictions: list[str],
    ) -> dict[str, Any]:
        labels = sorted(set(references) | set(predictions))
        accuracy = self._ratio(
            sum(
                1
                for reference, prediction in zip(references, predictions, strict=True)
                if reference == prediction
            ),
            len(references),
        )
        precision_values = []
        recall_values = []
        f1_values = []
        per_class: dict[str, dict[str, float]] = {}
        for label in labels:
            tp, fp, fn, _ = self._binary_counts(
                [value == label for value in references],
                [value == label for value in predictions],
            )
            precision = self._ratio(tp, tp + fp)
            recall = self._ratio(tp, tp + fn)
            f1 = self._f1(precision, recall)
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
            per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        return {
            "accuracy": accuracy,
            "macro_precision": self._mean(precision_values),
            "macro_recall": self._mean(recall_values),
            "macro_f1": self._mean(f1_values),
            "per_class": per_class,
            "confusion_matrix": self.confusion_matrix(references, predictions),
        }

    def field_metrics(
        self,
        references: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> dict[str, float]:
        true_positive = false_positive = false_negative = 0
        exact_matches = normalized_matches = reference_values = 0
        for reference, prediction in zip(references, predictions, strict=True):
            reference_norm = {key: self._normalize(value) for key, value in reference.items()}
            prediction_norm = {key: self._normalize(value) for key, value in prediction.items()}
            for key, predicted_value in prediction_norm.items():
                if key in reference_norm and reference_norm[key] == predicted_value:
                    true_positive += 1
                else:
                    false_positive += 1
            for key, reference_value in reference_norm.items():
                reference_values += 1
                if prediction_norm.get(key) == reference_value:
                    normalized_matches += 1
                else:
                    false_negative += 1
            exact_matches += int(reference_norm == prediction_norm)
        precision = self._ratio(true_positive, true_positive + false_positive)
        recall = self._ratio(true_positive, true_positive + false_negative)
        return {
            "field_precision": precision,
            "field_recall": recall,
            "field_f1": self._f1(precision, recall),
            "exact_match": self._ratio(exact_matches, len(references)),
            "normalized_value_accuracy": self._ratio(normalized_matches, reference_values),
        }

    def detection_metrics(
        self,
        references: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        reference_sets = [self._detections(item) for item in references]
        prediction_sets = [self._detections(item) for item in predictions]
        classes = sorted(
            {
                self._detection_label(item)
                for records in [*reference_sets, *prediction_sets]
                for item in records
            }
        )
        thresholds = [round(0.5 + index * 0.05, 2) for index in range(10)]
        ap_by_threshold: dict[float, list[float]] = defaultdict(list)
        per_class: dict[str, dict[str, float]] = {}
        matched_ious: list[float] = []
        for label in classes:
            total_references = sum(
                1
                for records in reference_sets
                for item in records
                if self._detection_label(item) == label
            )
            threshold_flags: dict[float, list[tuple[float, bool]]] = {
                threshold: [] for threshold in thresholds
            }
            for threshold in thresholds:
                for reference_records, predicted_records in zip(
                    reference_sets,
                    prediction_sets,
                    strict=True,
                ):
                    candidates = sorted(
                        [
                            item
                            for item in predicted_records
                            if self._detection_label(item) == label
                        ],
                        key=lambda item: float(item.get("confidence", 0.0)),
                        reverse=True,
                    )
                    eligible = [
                        item
                        for item in reference_records
                        if self._detection_label(item) == label
                    ]
                    matched: set[int] = set()
                    for candidate in candidates:
                        best = self._best_box_match(
                            candidate,
                            eligible,
                            matched,
                            require_label=False,
                        )
                        score = float(candidate.get("confidence", 0.0))
                        if best is not None and self._iou(candidate, eligible[best]) >= threshold:
                            matched.add(best)
                            threshold_flags[threshold].append((score, True))
                            if threshold == 0.5:
                                matched_ious.append(self._iou(candidate, eligible[best]))
                        else:
                            threshold_flags[threshold].append((score, False))
                ap_by_threshold[threshold].append(
                    self._average_precision(threshold_flags[threshold], total_references)
                )
            flags50 = threshold_flags[0.5]
            true_positive = sum(1 for _, matched in flags50 if matched)
            false_positive = len(flags50) - true_positive
            false_negative = max(0, total_references - true_positive)
            per_class[label] = {
                "precision": self._ratio(true_positive, true_positive + false_positive),
                "recall": self._ratio(true_positive, true_positive + false_negative),
                "ap50": self._average_precision(flags50, total_references),
            }
        map50 = self._mean(ap_by_threshold[0.5])
        map50_95 = self._mean(
            [value for values in ap_by_threshold.values() for value in values]
        )
        return {
            "mean_iou": self._mean(matched_ious),
            "map50": map50,
            "map50_95": map50_95,
            "per_class": per_class,
            "per_class_precision": {
                label: values["precision"] for label, values in per_class.items()
            },
            "per_class_recall": {
                label: values["recall"] for label, values in per_class.items()
            },
            "iou_thresholds": thresholds,
            "ap_method": "ranked all-points average precision",
        }

    def table_metrics(
        self,
        references: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> dict[str, float]:
        true_positive = false_positive = false_negative = 0
        structures = []
        teds_scores = []
        for reference, prediction in zip(references, predictions, strict=True):
            reference_tables = self._records(reference.get("tables"))
            prediction_tables = self._records(prediction.get("tables"))
            for index in range(max(len(reference_tables), len(prediction_tables))):
                reference_table = reference_tables[index] if index < len(reference_tables) else {}
                prediction_table = prediction_tables[index] if index < len(prediction_tables) else {}
                reference_cells = self._table_cells(reference_table)
                prediction_cells = self._table_cells(prediction_table)
                true_positive += len(reference_cells & prediction_cells)
                false_positive += len(prediction_cells - reference_cells)
                false_negative += len(reference_cells - prediction_cells)
                reference_shape = self._table_shape(reference_table)
                prediction_shape = self._table_shape(prediction_table)
                structures.append(float(reference_shape == prediction_shape))
                reference_serialized = self._table_serialized(reference_table)
                prediction_serialized = self._table_serialized(prediction_table)
                teds_scores.append(self._normalized_edit_similarity(
                    reference_serialized,
                    prediction_serialized,
                ))
        precision = self._ratio(true_positive, true_positive + false_positive)
        recall = self._ratio(true_positive, true_positive + false_negative)
        return {
            "cell_precision": precision,
            "cell_recall": recall,
            "cell_f1": self._f1(precision, recall),
            "structure_accuracy": self._mean(structures),
            "teds": self._mean(teds_scores),
        }

    def qa_metrics(self, references: list[Any], predictions: list[Any]) -> dict[str, float]:
        exact_values = []
        anls_values = []
        list_values = []
        no_answer_values = []
        for reference, prediction in zip(references, predictions, strict=True):
            reference_answers = self._answer_list(reference)
            prediction_answers = self._answer_list(prediction)
            reference_text = " | ".join(reference_answers)
            prediction_text = " | ".join(prediction_answers)
            exact_values.append(float(self._normalize(reference_text) == self._normalize(prediction_text)))
            anls_values.append(
                max(
                    (
                        self._anls(reference_answer, prediction_answer)
                        for reference_answer in reference_answers
                        for prediction_answer in prediction_answers
                    ),
                    default=0.0,
                )
            )
            list_values.append(self._list_anls(reference_answers, prediction_answers))
            if self._is_no_answer(reference_text):
                no_answer_values.append(float(self._is_no_answer(prediction_text)))
        return {
            "exact_match": self._mean(exact_values),
            "anls": self._mean(anls_values),
            "list_answer_anls": self._mean(list_values),
            "no_answer_accuracy": self._mean(no_answer_values),
            "no_answer_sample_count": len(no_answer_values),
        }

    def retrieval_metrics(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [item for item in records if item.get("relevant_ids") is not None]
        if not valid:
            return {"status": "not_calculated", "reason": "No relevance labels supplied."}
        requested_k = sorted(
            {
                int(value)
                for item in valid
                for value in item.get("k_values", [1, 3, 5, 10])
                if int(value) > 0
            }
        )
        recall_at_k: dict[str, float] = {}
        precision_at_k: dict[str, float] = {}
        ndcg_at_k: dict[str, float] = {}
        for k in requested_k:
            recalls = []
            precisions = []
            ndcgs = []
            for item in valid:
                relevant = {str(value) for value in item.get("relevant_ids", [])}
                retrieved = [str(value) for value in item.get("retrieved_ids", [])][:k]
                hits = sum(1 for value in retrieved if value in relevant)
                recalls.append(self._ratio(hits, len(relevant)))
                precisions.append(self._ratio(hits, k))
                dcg = sum(
                    (1.0 / math.log2(index + 2))
                    for index, value in enumerate(retrieved)
                    if value in relevant
                )
                ideal = sum(
                    1.0 / math.log2(index + 2)
                    for index in range(min(k, len(relevant)))
                )
                ndcgs.append(self._ratio(dcg, ideal))
            recall_at_k[str(k)] = self._mean(recalls)
            precision_at_k[str(k)] = self._mean(precisions)
            ndcg_at_k[str(k)] = self._mean(ndcgs)
        reciprocal_ranks = []
        average_precisions = []
        for item in valid:
            relevant = {str(value) for value in item.get("relevant_ids", [])}
            retrieved = [str(value) for value in item.get("retrieved_ids", [])]
            ranks = [index + 1 for index, value in enumerate(retrieved) if value in relevant]
            reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
            hits = 0
            precision_sum = 0.0
            for index, value in enumerate(retrieved, start=1):
                if value in relevant:
                    hits += 1
                    precision_sum += hits / index
            average_precisions.append(self._ratio(precision_sum, len(relevant)))
        return {
            "query_count": len(valid),
            "recall_at_k": recall_at_k,
            "precision_at_k": precision_at_k,
            "mrr": self._mean(reciprocal_ranks),
            "ndcg_at_k": ndcg_at_k,
            "evidence_document_map": self._mean(average_precisions),
        }

    def rag_metrics(self, judgements: list[dict[str, Any]]) -> dict[str, float | int]:
        def judged_mean(name: str) -> float:
            values = [self._unit(item[name]) for item in judgements if name in item]
            return self._mean(values)

        supported = sum(int(item.get("supported_claims", 0)) for item in judgements)
        total = sum(int(item.get("total_claims", 0)) for item in judgements)
        unsupported = sum(
            int(item.get("unsupported_claims", max(0, int(item.get("total_claims", 0)) - int(item.get("supported_claims", 0)))))
            for item in judgements
        )
        hallucinated = sum(int(item.get("hallucinated_claims", 0)) for item in judgements)
        return {
            "judgement_count": len(judgements),
            "faithfulness": judged_mean("faithfulness"),
            "answer_relevance": judged_mean("answer_relevance"),
            "context_relevance": judged_mean("context_relevance"),
            "citation_correctness": judged_mean("citation_correctness"),
            "unsupported_claim_rate": self._ratio(unsupported, total),
            "hallucination_rate": self._ratio(hallucinated, total),
            "supported_claim_count": supported,
            "total_claim_count": total,
        }

    def risk_metrics(
        self,
        records: list[dict[str, Any]],
        review_capacity: float,
    ) -> dict[str, float | int]:
        labels = [bool(item.get("label", False)) for item in records]
        scores = [float(item.get("score", item.get("prediction", 0.0))) for item in records]
        threshold = float(records[0].get("threshold", 0.5)) if records else 0.5
        predictions = [score >= threshold for score in scores]
        tp, fp, fn, tn = self._binary_counts(labels, predictions)
        precision = self._ratio(tp, tp + fp)
        recall = self._ratio(tp, tp + fn)
        capacity = max(0.0, min(1.0, review_capacity))
        review_count = math.ceil(len(records) * capacity)
        ranked = sorted(zip(scores, labels, strict=True), reverse=True)
        reviewed_positive = sum(1 for _, label in ranked[:review_count] if label)
        return {
            "sample_count": len(records),
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": self._f1(precision, recall),
            "auprc": self._binary_average_precision(scores, labels),
            "false_positive_rate": self._ratio(fp, fp + tn),
            "review_capacity": capacity,
            "recall_at_fixed_review_budget": self._ratio(
                reviewed_positive,
                sum(labels),
            ),
        }

    def confidence_metrics(
        self,
        records: list[dict[str, Any]],
        bins: int = 10,
    ) -> dict[str, Any]:
        samples = [
            (self._unit(item.get("confidence", 0.0)), bool(item.get("correct", False)))
            for item in records
        ]
        bins = max(2, min(100, bins))
        expected_calibration_error = 0.0
        calibration_bins = []
        for index in range(bins):
            lower = index / bins
            upper = (index + 1) / bins
            values = [
                (confidence, correct)
                for confidence, correct in samples
                if lower <= confidence < upper or (index == bins - 1 and confidence == 1.0)
            ]
            if not values:
                continue
            average_confidence = mean(confidence for confidence, _ in values)
            accuracy = mean(float(correct) for _, correct in values)
            weight = len(values) / len(samples)
            expected_calibration_error += weight * abs(accuracy - average_confidence)
            calibration_bins.append(
                {
                    "lower": self._round(lower),
                    "upper": self._round(upper),
                    "count": len(values),
                    "mean_confidence": self._round(average_confidence),
                    "accuracy": self._round(accuracy),
                }
            )
        ranked = sorted(samples, key=lambda item: item[0], reverse=True)
        curve = []
        errors = 0
        for index, (confidence, correct) in enumerate(ranked, start=1):
            errors += int(not correct)
            curve.append(
                {
                    "coverage": self._round(index / len(ranked)),
                    "risk": self._round(errors / index),
                    "minimum_confidence": self._round(confidence),
                }
            )
        aurc = 0.0
        previous_coverage = 0.0
        previous_risk = 0.0
        for point in curve:
            coverage = float(point["coverage"])
            risk = float(point["risk"])
            aurc += (coverage - previous_coverage) * (risk + previous_risk) / 2
            previous_coverage = coverage
            previous_risk = risk
        return {
            "sample_count": len(samples),
            "expected_calibration_error": self._round(expected_calibration_error),
            "aurc": self._round(aurc),
            "risk_coverage_curve": curve,
            "calibration_bins": calibration_bins,
        }

    def human_review_metrics(self, records: list[dict[str, Any]]) -> dict[str, float | int]:
        total_fields = sum(int(item.get("field_count", 0)) for item in records)
        accepted = sum(int(item.get("accepted_fields", 0)) for item in records)
        corrected = sum(int(item.get("corrected_fields", 0)) for item in records)
        review_times = [float(item["review_time_seconds"]) for item in records if "review_time_seconds" in item]
        grouped_labels: dict[str, list[str]] = defaultdict(list)
        for item in records:
            if item.get("document_id") and item.get("reviewer_label") is not None:
                grouped_labels[str(item["document_id"])].append(str(item["reviewer_label"]))
        agreements = []
        for labels in grouped_labels.values():
            if len(labels) < 2:
                continue
            pairs = 0
            equal = 0
            for left in range(len(labels)):
                for right in range(left + 1, len(labels)):
                    pairs += 1
                    equal += int(labels[left] == labels[right])
            agreements.append(self._ratio(equal, pairs))
        critical_total = sum(int(item.get("critical_error_count", 0)) for item in records)
        critical_escaped = sum(int(item.get("critical_error_escaped", 0)) for item in records)
        return {
            "review_count": len(records),
            "acceptance_rate": self._ratio(accepted, total_fields),
            "correction_rate": self._ratio(corrected, total_fields),
            "mean_review_time_seconds": self._mean(review_times),
            "escalation_rate": self._ratio(
                sum(bool(item.get("escalated", False)) for item in records),
                len(records),
            ),
            "reviewer_agreement": self._mean(agreements),
            "critical_error_escape_rate": self._ratio(critical_escaped, critical_total),
        }

    def system_metrics(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        fields = {
            "latency_per_page": "latency_per_page_seconds",
            "full_document_latency": "full_document_latency_seconds",
            "throughput": "pages_per_second",
            "peak_memory": "peak_memory_mb",
            "queue_time": "queue_time_seconds",
            "retrieval_latency": "retrieval_latency_seconds",
        }
        output: dict[str, Any] = {"measurement_count": len(records)}
        for output_name, input_name in fields.items():
            values = sorted(float(item[input_name]) for item in records if input_name in item)
            output[output_name] = {
                "mean": self._mean(values),
                "p95": self._percentile(values, 0.95),
                "samples": len(values),
            }
        output["failure_rate"] = self._ratio(
            sum(bool(item.get("failed", False)) for item in records),
            len(records),
        )
        return output

    def character_error_rate(self, reference: str, prediction: str) -> float:
        return self._round(
            self._levenshtein_sequence(list(reference), list(prediction))
            / max(len(reference), 1)
        )

    def word_error_rate(self, reference: str, prediction: str) -> float:
        reference_words = reference.split()
        prediction_words = prediction.split()
        return self._round(
            self._levenshtein_sequence(reference_words, prediction_words)
            / max(len(reference_words), 1)
        )

    def confusion_matrix(
        self,
        references: list[str],
        predictions: list[str],
    ) -> dict[str, dict[str, int]]:
        labels = sorted(set(references) | set(predictions))
        matrix = {label: Counter() for label in labels}
        for reference, prediction in zip(references, predictions, strict=True):
            matrix[reference][prediction] += 1
        return {
            label: {predicted: int(counts.get(predicted, 0)) for predicted in labels}
            for label, counts in matrix.items()
        }

    def _not_calculated(self, reason: str) -> dict[str, Any]:
        return {
            "status": "not_calculated",
            "reason": reason,
            "fake_metrics_reported": False,
            "supported_metrics": self.supported_metrics(),
        }

    def _dataset_provenance(self, value: Any) -> dict[str, Any]:
        dataset = self._mapping(value)
        return {
            "name": dataset.get("name", "user_supplied_labels"),
            "version": dataset.get("version"),
            "split": dataset.get("split", "unspecified"),
            "provenance": dataset.get("provenance", "direct API payload"),
            "document_split": dataset.get("document_split", True),
        }

    def _digest(self, payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _has_any(self, records: list[dict[str, Any]], key: str) -> bool:
        return any(key in item and item.get(key) is not None for item in records)

    def _has_detection_data(self, records: list[dict[str, Any]]) -> bool:
        return any(self._detections(item) for item in records)

    def _detections(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        values = []
        for key in ("layout_regions", "detections", "visual_detections"):
            values.extend(self._records(record.get(key)))
        return values

    def _detection_label(self, record: dict[str, Any]) -> str:
        return str(
            record.get("region_type")
            or record.get("detection_class")
            or record.get("class")
            or record.get("label")
            or "object"
        )

    def _bbox(self, record: dict[str, Any]) -> tuple[float, float, float, float] | None:
        value = record.get("bounding_box", record.get("bbox"))
        if isinstance(value, dict):
            try:
                return (
                    float(value["x1"]),
                    float(value["y1"]),
                    float(value["x2"]),
                    float(value["y2"]),
                )
            except (KeyError, TypeError, ValueError):
                return None
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                return tuple(float(item) for item in value)  # type: ignore[return-value]
            except (TypeError, ValueError):
                return None
        return None

    def _iou(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        left_box = self._bbox(left)
        right_box = self._bbox(right)
        if not left_box or not right_box:
            return 0.0
        x1 = max(left_box[0], right_box[0])
        y1 = max(left_box[1], right_box[1])
        x2 = min(left_box[2], right_box[2])
        y2 = min(left_box[3], right_box[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        left_area = max(0.0, left_box[2] - left_box[0]) * max(0.0, left_box[3] - left_box[1])
        right_area = max(0.0, right_box[2] - right_box[0]) * max(0.0, right_box[3] - right_box[1])
        return self._ratio(intersection, left_area + right_area - intersection)

    def _best_box_match(
        self,
        candidate: dict[str, Any],
        references: list[dict[str, Any]],
        matched: set[int],
        *,
        require_label: bool,
    ) -> int | None:
        options = [
            (self._iou(candidate, reference), index)
            for index, reference in enumerate(references)
            if index not in matched
            and (
                not require_label
                or self._detection_label(candidate) == self._detection_label(reference)
            )
        ]
        return max(options)[1] if options else None

    def _average_precision(
        self,
        flags: list[tuple[float, bool]],
        total_references: int,
    ) -> float:
        if total_references <= 0:
            return 0.0
        true_positive = 0
        precision_sum = 0.0
        for rank, (_, matched) in enumerate(sorted(flags, reverse=True), start=1):
            if matched:
                true_positive += 1
                precision_sum += true_positive / rank
        return self._round(precision_sum / total_references)

    def _table_cells(self, table: dict[str, Any]) -> set[tuple[int, int, str]]:
        cells = self._records(table.get("cells"))
        if cells:
            return {
                (
                    int(cell.get("row_index", 0)),
                    int(cell.get("column_index", 0)),
                    self._normalize(cell.get("text", "")),
                )
                for cell in cells
            }
        rows = table.get("rows") or []
        return {
            (row_index, column_index, self._normalize(value))
            for row_index, row in enumerate(rows)
            for column_index, value in enumerate(row)
        }

    def _table_shape(self, table: dict[str, Any]) -> tuple[int, tuple[int, ...]]:
        rows = table.get("rows") or []
        if rows:
            return len(rows), tuple(len(row) for row in rows)
        cells = self._records(table.get("cells"))
        if not cells:
            return 0, ()
        row_counts = Counter(int(cell.get("row_index", 0)) for cell in cells)
        return len(row_counts), tuple(row_counts[index] for index in sorted(row_counts))

    def _table_serialized(self, table: dict[str, Any]) -> str:
        if table.get("html"):
            return re.sub(r"\s+", " ", str(table["html"]).strip().lower())
        return json.dumps(table.get("rows") or [], separators=(",", ":"), default=str).lower()

    def _answer_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value] or [""]
        text = str(value)
        if "|" in text or ";" in text:
            return [item.strip() for item in re.split(r"[|;]", text) if item.strip()] or [""]
        return [text]

    def _anls(self, reference: str, prediction: str) -> float:
        similarity = self._normalized_edit_similarity(
            self._normalize(reference),
            self._normalize(prediction),
        )
        return similarity if similarity >= 0.5 else 0.0

    def _list_anls(self, references: list[str], predictions: list[str]) -> float:
        if not references:
            return float(not predictions)
        available = set(range(len(predictions)))
        scores = []
        for reference in references:
            options = [
                (self._anls(reference, predictions[index]), index)
                for index in available
            ]
            if not options:
                scores.append(0.0)
                continue
            score, index = max(options)
            available.remove(index)
            scores.append(score)
        return self._mean(scores)

    def _is_no_answer(self, value: str) -> bool:
        normalized = self._normalize(value)
        return normalized in {"", "not found", "no answer", NO_ANSWER_TEXT}

    def _normalized_edit_similarity(self, reference: str, prediction: str) -> float:
        maximum = max(len(reference), len(prediction), 1)
        return self._round(
            1.0 - self._levenshtein_sequence(list(reference), list(prediction)) / maximum
        )

    def _binary_average_precision(self, scores: list[float], labels: list[bool]) -> float:
        positives = sum(labels)
        if not positives:
            return 0.0
        hits = 0
        precision_sum = 0.0
        for rank, (_, label) in enumerate(
            sorted(zip(scores, labels, strict=True), reverse=True),
            start=1,
        ):
            if label:
                hits += 1
                precision_sum += hits / rank
        return self._round(precision_sum / positives)

    def _binary_counts(
        self,
        labels: list[bool],
        predictions: list[bool],
    ) -> tuple[int, int, int, int]:
        true_positive = false_positive = false_negative = true_negative = 0
        for label, prediction in zip(labels, predictions, strict=True):
            true_positive += int(label and prediction)
            false_positive += int(not label and prediction)
            false_negative += int(label and not prediction)
            true_negative += int(not label and not prediction)
        return true_positive, false_positive, false_negative, true_negative

    def _levenshtein_sequence(
        self,
        reference: list[Any],
        prediction: list[Any],
    ) -> int:
        previous = list(range(len(prediction) + 1))
        for index, reference_item in enumerate(reference, start=1):
            current = [index]
            for prediction_index, prediction_item in enumerate(prediction, start=1):
                insert = current[prediction_index - 1] + 1
                delete = previous[prediction_index] + 1
                substitute = previous[prediction_index - 1] + (
                    reference_item != prediction_item
                )
                current.append(min(insert, delete, substitute))
            previous = current
        return previous[-1]

    def _mapping(self, value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    def _records(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [dict(item) for item in value if isinstance(item, dict)]

    def _normalize(self, value: Any) -> str:
        return " ".join(str(value).strip().lower().split())

    def _unit(self, value: Any) -> float:
        if isinstance(value, bool):
            return float(value)
        return max(0.0, min(1.0, float(value)))

    def _f1(self, precision: float, recall: float) -> float:
        return self._round(2 * precision * recall / max(precision + recall, 1e-12))

    def _ratio(self, numerator: float, denominator: float) -> float:
        return self._round(float(numerator) / float(denominator)) if denominator else 0.0

    def _mean(self, values: Iterable[float]) -> float:
        materialized = list(values)
        return self._round(mean(materialized)) if materialized else 0.0

    def _round(self, value: float) -> float:
        return round(float(value), 4)

    def _percentile(self, values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, max(0, math.ceil(len(values) * fraction) - 1))
        return self._round(values[index])


vision_evaluation_service = VisionEvaluationService()
