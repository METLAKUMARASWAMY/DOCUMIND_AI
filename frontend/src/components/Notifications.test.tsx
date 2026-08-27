import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { NotificationProvider, useNotify } from "./Notifications";

function Trigger({ tone, message }: { tone: "success" | "error"; message: string }) {
  const notify = useNotify();
  return (
    <button type="button" onClick={() => notify(tone, message)}>
      fire
    </button>
  );
}

describe("NotificationProvider", () => {
  it("shows a message after an action reports success", async () => {
    render(
      <NotificationProvider>
        <Trigger tone="success" message="Review approved" />
      </NotificationProvider>,
    );

    await userEvent.click(screen.getByRole("button", { name: "fire" }));
    expect(await screen.findByText("Review approved")).toBeTruthy();
  });

  it("lets an error be dismissed rather than expiring on its own", async () => {
    render(
      <NotificationProvider>
        <Trigger tone="error" message="Upload rejected" />
      </NotificationProvider>,
    );

    await userEvent.click(screen.getByRole("button", { name: "fire" }));
    expect(await screen.findByText("Upload rejected")).toBeTruthy();

    await userEvent.click(screen.getByRole("button", { name: /close/i }));
    await waitFor(() => expect(screen.queryByText("Upload rejected")).toBeNull());
  });

  it("is a no-op outside the provider so components stay independently testable", async () => {
    render(<Trigger tone="success" message="Nothing mounted" />);
    await userEvent.click(screen.getByRole("button", { name: "fire" }));
    expect(screen.queryByText("Nothing mounted")).toBeNull();
  });
});
