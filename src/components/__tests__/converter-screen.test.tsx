import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConverterScreen, DownloadFailureDetails } from "../converter-screen";

const useConverterMock = vi.hoisted(() => vi.fn());

vi.mock("@/hooks/use-converter", () => ({
  useConverter: useConverterMock,
}));

vi.mock("@tauri-apps/api/webview", () => ({
  getCurrentWebview: () => ({
    onDragDropEvent: vi.fn().mockResolvedValue(() => {}),
  }),
}));

afterEach(() => {
  vi.useRealTimers();
});

describe("DownloadFailureDetails", () => {
  it("announces copied diagnostics and resets after two seconds", async () => {
    vi.useFakeTimers();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    render(
      <DownloadFailureDetails
        message={{
          primary: "2 images were skipped. See details below.",
          breakdown: [
            "1 download link expired — one.jpg",
            "1 image was not found — two.jpg",
          ],
          diagnostics: "expired_url: 1; one.jpg\nnot_found: 1; HTTP 404; two.jpg",
        }}
      />,
    );

    expect(screen.getByText("Download details")).toBeInTheDocument();
    expect(screen.getByText("1 download link expired — one.jpg")).toBeInTheDocument();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy details" }));
      await Promise.resolve();
    });
    expect(writeText).toHaveBeenCalledWith(
      "expired_url: 1; one.jpg\nnot_found: 1; HTTP 404; two.jpg",
    );
    expect(screen.getByRole("status")).toHaveTextContent("Copied");

    act(() => {
      vi.advanceTimersByTime(2_000);
    });

    expect(screen.queryByText("Copied")).not.toBeInTheDocument();
  });

  it("clears copied reset timer on unmount", async () => {
    vi.useFakeTimers();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
    const clearTimeoutSpy = vi.spyOn(window, "clearTimeout");
    const { unmount } = render(
      <DownloadFailureDetails
        message={{
          primary: "1 image was skipped.",
          breakdown: ["1 download failed. Try again."],
          diagnostics: "download_error: 1",
        }}
      />,
    );

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy details" }));
      await Promise.resolve();
    });
    expect(screen.getByRole("status")).toHaveTextContent("Copied");

    unmount();

    expect(clearTimeoutSpy).toHaveBeenCalled();
    clearTimeoutSpy.mockRestore();
  });

  it("reports clipboard failures", async () => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: vi.fn().mockRejectedValue(new Error("denied")),
      },
    });

    render(
      <DownloadFailureDetails
        message={{
          primary: "1 image was skipped.",
          breakdown: ["1 download failed. Try again."],
          diagnostics: "download_error: 1",
        }}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Copy details" }));

    expect(
      await screen.findByText("Couldn't copy details"),
    ).toBeInTheDocument();
  });

  it("renders structured hard failures through the error card", () => {
    useConverterMock.mockReturnValue({
      selectedFile: "/tmp/dataset.ndjson",
      selectedFileName: "dataset.ndjson",
      selectedFormat: null,
      setSelectedFormat: vi.fn(),
      isConverting: false,
      progress: null,
      result: null,
      error: {
        kind: "download_failed",
        message: "Images could not be downloaded.",
        failure_summary: {
          groups: [
            {
              kind: "expired_url",
              count: 1,
              examples: ["one.jpg"],
              http_statuses: [],
            },
          ],
          expiry: {
            all_expired: true,
            latest_expired_at: 1_784_419_200,
          },
        },
      },
      elapsedSeconds: 0,
      selectFile: vi.fn(),
      removeFile: vi.fn(),
      setFileFromPath: vi.fn(),
      handleConvert: vi.fn(),
      resetState: vi.fn(),
      getProgressPercentage: vi.fn(),
      formatElapsedTime: vi.fn(),
      getDownloadRate: vi.fn(),
    });

    render(<ConverterScreen onBack={vi.fn()} />);

    expect(screen.getByTestId("error-message")).toHaveTextContent(
      "Your download links expired on 19 Jul 2026.",
    );
    expect(screen.getByText("Download details")).toBeInTheDocument();
  });
});
