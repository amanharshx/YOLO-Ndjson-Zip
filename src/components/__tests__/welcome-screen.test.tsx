import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { WelcomeScreen } from "../welcome-screen";

vi.mock("../update-checker", () => ({ UpdateChecker: () => null }));

describe("WelcomeScreen", () => {
  it("lists depth among the supported task types", () => {
    render(<WelcomeScreen onGetStarted={() => {}} />);

    expect(screen.getByText("Depth")).toBeInTheDocument();
    expect(screen.getByText(/formats across/).parentElement).toHaveTextContent(
      "12 formats across 7 task types",
    );
  });
});
