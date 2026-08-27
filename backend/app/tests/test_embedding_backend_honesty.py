"""What the retrieval backend is, and what it is not, must both be stated.

Directive: "Do not allow a UI statement like 'BGE-M3 active' unless BGE-M3 is
genuinely generating embeddings in the live system. If the current production
retrieval path is local hashing: clearly label it as the current retrieval
backend, expose its identity honestly."

The audit measured the difference rather than assuming it. Cosine similarity
between a paraphrase and the field that answers it:

                                          BGE-M3    feature hashing
    "how much rent is payable each month"
        -> "Monthly Rent: 1,275.00 GBP"    0.7070    0.0925
    "who owns the property being let"
        -> "Landlord: Brackenfield"        0.5448    0.0000

Feature hashing scores exact-token overlap. It is a real, deterministic,
dependency-free backend and it is what ships by default — it simply cannot
match meaning, and the system must not imply otherwise.
"""

from __future__ import annotations

import importlib.util

import pytest

from app.core.config import settings
from app.services.vision.retrieval_adapters import (
    BGEM3EmbeddingAdapter,
    LocalHashingEmbeddingAdapter,
    LocalHybridRetrievalIndex,
)


@pytest.fixture(scope="module")
def index() -> LocalHybridRetrievalIndex:
    return LocalHybridRetrievalIndex()


def test_the_active_embedding_backend_names_itself(index):
    status = index.status()

    assert status["embedding_backend"] == status["embedding"]["adapter_id"]
    assert status["embedding_model"] == status["embedding"]["model_version"]
    assert status["embedding_backend"], "the backend must be named, not implied"


def test_a_hashing_backend_is_never_reported_as_semantic(index):
    status = index.status()

    if status["embedding_backend"] == "local-hashing-embedding":
        assert status["semantic_embeddings"] is False
        assert "hashing" in status["embedding_model"] or "hash" in status["embedding_backend"]


def test_the_pipeline_description_does_not_claim_a_stage_that_is_disabled(index):
    """It used to be the fixed string "top30_hybrid_then_top15_rerank_then_top5_to_8",
    which named a rerank stage even when the reranker was disabled."""
    status = index.status()
    pipeline = status["pipeline"]

    if status["reranker"]["status"] != "active":
        assert "top15_rerank" not in pipeline, pipeline
        assert "no_rerank" in pipeline, pipeline
    else:
        assert "rerank" in pipeline, pipeline


def test_the_pipeline_description_names_the_embedding_model(index):
    status = index.status()
    assert status["embedding_model"] in status["pipeline"], status["pipeline"]


def test_bge_m3_reports_unavailable_rather_than_active_when_it_cannot_run(monkeypatch):
    """A model that cannot initialise must never read as ACTIVE."""
    adapter = BGEM3EmbeddingAdapter()
    monkeypatch.setattr(settings, "enable_heavy_models", True)
    monkeypatch.setattr(settings, "enable_bge_m3", True)
    monkeypatch.setattr(settings, "bge_allow_download", False)
    monkeypatch.setattr(settings, "bge_m3_checkpoint", "BAAI/definitely-not-a-real-checkpoint")

    status = adapter.status()
    assert status.status != "active"
    assert status.reason


def test_bge_m3_reports_disabled_when_the_flag_is_off(monkeypatch):
    adapter = BGEM3EmbeddingAdapter()
    monkeypatch.setattr(settings, "enable_bge_m3", False)

    assert adapter.status().status == "disabled"


def test_feature_hashing_cannot_match_a_paraphrase(monkeypatch):
    """The measured limitation, pinned so it cannot be quietly overstated."""
    adapter = LocalHashingEmbeddingAdapter()

    def cosine(left: str, right: str) -> float:
        a = adapter.embed(left).dense
        b = adapter.embed(right).dense
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    # Shares the token "rent": a weak but non-zero score.
    with_overlap = cosine("how much rent is payable each month", "Monthly Rent: 1,275.00 GBP")
    # Shares no token at all: exactly nothing.
    without_overlap = cosine("who owns the property being let", "Landlord: Brackenfield Estates")

    assert without_overlap == pytest.approx(0.0, abs=1e-6), without_overlap
    assert with_overlap < 0.2, with_overlap


@pytest.mark.skipif(
    importlib.util.find_spec("FlagEmbedding") is None,
    reason="FlagEmbedding is not installed in this interpreter",
)
def test_bge_m3_resolves_the_paraphrases_that_hashing_cannot(monkeypatch):
    """Runs only where the optional backend is genuinely installed.

    This is the activation path, executed rather than asserted. Where it cannot
    run it skips loudly instead of passing vacuously.
    """
    monkeypatch.setattr(settings, "enable_heavy_models", True)
    monkeypatch.setattr(settings, "enable_bge_m3", True)

    adapter = BGEM3EmbeddingAdapter()
    if adapter.status().status != "active":
        pytest.skip(f"BGE-M3 checkpoint unavailable: {adapter.status().reason}")

    vectors = adapter.embed_many(
        [
            "Monthly Rent: 1,275.00 GBP",
            "Landlord: Brackenfield Estates",
            "Port of Discharge: Santos",
            "how much rent is payable each month",
            "who owns the property being let",
        ]
    )
    assert len(vectors[0].dense) == adapter.dimensions

    def cosine(i: int, j: int) -> float:
        a, b = vectors[i].dense, vectors[j].dense
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb)

    assert cosine(3, 0) > cosine(3, 2), "rent paraphrase did not outrank an unrelated chunk"
    assert cosine(4, 1) > cosine(4, 0), "landlord paraphrase did not outrank an unrelated chunk"
