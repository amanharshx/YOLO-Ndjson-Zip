import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useConverter } from "../use-converter";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  Channel: vi.fn(() => ({ onmessage: null })),
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn(),
  save: vi.fn(),
}));

describe("useConverter", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("setFileFromPath sets file and extracts filename", () => {
    const { result } = renderHook(() => useConverter());

    act(() => {
      result.current.setFileFromPath("/path/to/dataset.ndjson");
    });

    expect(result.current.selectedFile).toBe("/path/to/dataset.ndjson");
    expect(result.current.selectedFileName).toBe("dataset.ndjson");
  });

  it("removeFile clears file state", () => {
    const { result } = renderHook(() => useConverter());

    act(() => {
      result.current.setFileFromPath("/path/to/dataset.ndjson");
    });

    expect(result.current.selectedFile).toBe("/path/to/dataset.ndjson");

    act(() => {
      result.current.removeFile();
    });

    expect(result.current.selectedFile).toBeNull();
    expect(result.current.selectedFileName).toBeNull();
  });

  it("getProgressPercentage calculates correctly", () => {
    const { result } = renderHook(() => useConverter());

    // With no progress, should return 0
    expect(result.current.getProgressPercentage()).toBe(0);
  });

  it("formatElapsedTime formats as M:SS", () => {
    const { result } = renderHook(() => useConverter());

    expect(result.current.formatElapsedTime(0)).toBe("0:00");
    expect(result.current.formatElapsedTime(5)).toBe("0:05");
    expect(result.current.formatElapsedTime(65)).toBe("1:05");
    expect(result.current.formatElapsedTime(125)).toBe("2:05");
  });

  it("getDownloadRate returns rate during download phase", () => {
    const { result } = renderHook(() => useConverter());

    // With no progress or elapsed time, should return "0.0"
    expect(result.current.getDownloadRate()).toBe("0.0");
  });
});
