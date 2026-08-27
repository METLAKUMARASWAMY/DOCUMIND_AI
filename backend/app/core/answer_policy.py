"""Wording the system uses when it cannot answer.

This lives in `app.core` rather than beside the service that emits it because
`app.services.vision.__init__` instantiates the whole VisionAIService on
import. Anything importing a constant from inside that package drags the
service in with it — which is how a circular import appeared between the model
registry and the service the first time this constant was shared.
"""

from __future__ import annotations

# The wording of an abstention is a truth claim in itself. "The documents do
# not contain sufficient evidence" asserts something about the documents; what
# the system actually knows is that nothing *matched*. Those differ whenever
# retrieval is the limiting factor — with the feature-hashing embedding
# backend, a paraphrase sharing no tokens with a field label scores 0.0000
# against it, so evidence that plainly exists is never reached. Saying
# "matched" keeps the claim inside what was observed.
UNSUPPORTED_ANSWER = "No evidence in the selected documents matched this question."

__all__ = ["UNSUPPORTED_ANSWER"]
