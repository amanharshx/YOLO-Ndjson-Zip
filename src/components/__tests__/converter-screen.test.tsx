import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  ConverterScreen,
  DownloadFailureDetails,
  getDownloadMessage,
} from "../converter-screen";
import type { ConvertResult } from "@/lib/types";

const useConverterMock = vi.hoisted(() => vi.fn());

vi.mock("@/hooks/use-converter", () => ({
  useConverter: useConverterMock,
}));

vi.mock("@tauri-apps/api/webview", () => ({
  getCurrentWebview: () => ({
    onDragDropEvent: vi.fn().mockResolvedValue(() => {}),
  }),
}));

const completeResult: ConvertResult = {
  zip_path: "/tmp/dataset.zip",
  file_count: 4,
  image_count: 2,
  download_total: 2,
  failed_downloads: 0,
  omitted_images: 0,
  expired_url_failures: 0,
  failure_summary: {
    groups: [],
    expiry: null,
  },
};

describe("getDownloadMessage", () => {
  it("returns no warning when conversion is complete", () => {
    expect(getDownloadMessage(completeResult)).toBeNull();
  });

  it("uses skipped vocabulary and structured causes", () => {
    expect(
      getDownloadMessage({
        ...completeResult,
        download_total: 2,
        failed_downloads: 2,
        omitted_images: 2,
        expired_url_failures: 2,
        failure_summary: {
          groups: [
            {
              kind: "expired_url",
              count: 2,
              examples: ["one.jpg", "two.jpg"],
              http_statuses: [],
            },
          ],
          expiry: null,
        },
      })?.primary,
    ).toBe(
      "2 images and their corresponding annotations were skipped. Their download links expired. Export the dataset again from Ultralytics Platform, then retry.",
    );
  });
});

describe("DownloadFailureDetails", () => {
  it("renders one collapsed disclosure and copies diagnostics", () => {
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
    fireEvent.click(screen.getByRole("button", { name: "Copy details" }));
    expect(writeText).toHaveBeenCalledWith(
      "expired_url: 1; one.jpg\nnot_found: 1; HTTP 404; two.jpg",
    );
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
            urls_with_expiry: 1,
            expired_urls: 1,
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

    expect(screen.getByTestId("error-card")).toHaveTextContent(
      "Your download links expired on 19 Jul 2026.",
    );
    expect(screen.getByText("Download details")).toBeInTheDocument();
  });
});
