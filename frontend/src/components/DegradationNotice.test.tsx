import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { DegradationNotice, type AnswerDegradation } from "./DegradationNotice";

/**
 * A degradation the user never sees is still a silent fallback. The backend
 * discloses on VisionAskResponse.degradations; this component is the half that
 * makes the disclosure real.
 */

const generatorFailed: AnswerDegradation = {
  component: "qwen-local",
  stage: "answer_generation",
  detail: "The grounded generator raised RuntimeError and was skipped.",
  impact: "generator_skipped",
};

const answerRejected: AnswerDegradation = {
  component: "grounding_validator",
  stage: "grounding_validation",
  detail: "The generated answer cited no evidence that resolves to the retrieved set.",
  impact: "generated_answer_rejected",
};

describe("DegradationNotice", () => {
  it("renders nothing when the answer was not degraded", () => {
    const { container } = render(<DegradationNotice degradations={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the field is absent", () => {
    const { container } = render(<DegradationNotice />);
    expect(container).toBeEmptyDOMElement();
  });

  it("tolerates null without throwing", () => {
    const { container } = render(<DegradationNotice degradations={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("warns when a generator was skipped", () => {
    render(<DegradationNotice degradations={[generatorFailed]} />);

    expect(screen.getByText(/reduced capability/i)).toBeTruthy();
    const { textContent } = document.body;
    expect(textContent).toContain("Answer generation");
    expect(textContent).toContain("RuntimeError");
  });

  it("reports a rejected ungrounded answer as reassurance, not alarm", () => {
    render(<DegradationNotice degradations={[answerRejected]} />);

    // The grounding contract working is good news for the reader.
    expect(screen.getByText(/ungrounded answer was rejected/i)).toBeTruthy();
    expect(document.body.textContent).toContain("Grounding validation");
  });

  it("lists every degradation when more than one occurred", () => {
    const { container } = render(
      <DegradationNotice degradations={[generatorFailed, answerRejected]} />,
    );

    expect(container.querySelectorAll(".degradation-list li")).toHaveLength(2);
    expect(container.textContent).toContain("Answer generation");
    expect(container.textContent).toContain("Grounding validation");
  });

  it("falls back to a readable label for an unmapped stage", () => {
    render(<DegradationNotice degradations={[{ ...generatorFailed, stage: "some_new_stage" }]} />);

    expect(document.body.textContent).toContain("some new stage");
  });
});
