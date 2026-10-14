import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { WelcomeScreen } from "../welcome-screen";

function renderWelcome(updateAvailable = false) {
  const onOpenAbout = vi.fn();
  render(
    <WelcomeScreen
      onGetStarted={() => {}}
      onOpenAbout={onOpenAbout}
      updateAvailable={updateAvailable}
    />,
  );
  return onOpenAbout;
}

describe("WelcomeScreen", () => {
  it("lists depth among the supported task types", () => {
    renderWelcome();

    expect(screen.getByText("Depth")).toBeInTheDocument();
    expect(screen.getByText(/formats across/).parentElement).toHaveTextContent(
      "12 formats across 7 task types",
    );
  });

  it("opens About from the About button", () => {
    const onOpenAbout = renderWelcome();

    fireEvent.click(screen.getByRole("button", { name: /About/ }));

    expect(onOpenAbout).toHaveBeenCalledOnce();
    expect(screen.queryByLabelText("Update available")).not.toBeInTheDocument();
  });

  it("marks the About button when an update is available", () => {
    renderWelcome(true);

    expect(screen.getByLabelText("Update available")).toBeInTheDocument();
  });
});
