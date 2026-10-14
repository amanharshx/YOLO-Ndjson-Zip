import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { check, type DownloadEvent, type Update } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";
import { useUpdater } from "../use-updater";

vi.mock("@tauri-apps/plugin-updater", () => ({ check: vi.fn() }));
vi.mock("@tauri-apps/plugin-process", () => ({ relaunch: vi.fn() }));

const checkMock = vi.mocked(check);

function fakeUpdate(downloadAndInstall = vi.fn()): Update {
  return {
    version: "9.9.9",
    date: "2026-10-01T00:00:00Z",
    body: "Release notes",
    downloadAndInstall,
  } as unknown as Update;
}

describe("useUpdater", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("silent check marks an available update without opening the dialog", async () => {
    checkMock.mockResolvedValue(fakeUpdate());
    const { result } = renderHook(() => useUpdater());

    await act(() => result.current.checkForUpdates({ silent: true }));

    expect(result.current.state).toBe("available");
    expect(checkMock).toHaveBeenCalledWith({ timeout: 10_000 });
    expect(result.current.dialogOpen).toBe(false);
    expect(result.current.release).toEqual({
      version: "9.9.9",
      date: "2026-10-01T00:00:00Z",
      notes: "Release notes",
    });
  });

  it("silent check stays idle when up to date or offline", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { result } = renderHook(() => useUpdater());

    checkMock.mockResolvedValue(null);
    await act(() => result.current.checkForUpdates({ silent: true }));
    expect(result.current.state).toBe("idle");

    checkMock.mockRejectedValue(new Error("offline"));
    await act(() => result.current.checkForUpdates({ silent: true }));
    expect(result.current.state).toBe("idle");
    expect(result.current.dialogOpen).toBe(false);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("manual check opens the dialog and reports up to date", async () => {
    checkMock.mockResolvedValue(null);
    const { result } = renderHook(() => useUpdater());

    await act(() => result.current.checkForUpdates());

    expect(result.current.dialogOpen).toBe(true);
    expect(result.current.state).toBe("up-to-date");
  });

  it("manual check surfaces the failure", async () => {
    checkMock.mockRejectedValue(new Error("rate limited"));
    const { result } = renderHook(() => useUpdater());

    await act(() => result.current.checkForUpdates());

    expect(result.current.state).toBe("error");
    expect(result.current.error).toBe("rate limited");
  });

  it("lets the dialog close mid-check and ignores a stale silent result", async () => {
    let resolveSilent: (update: Update | null) => void = () => {};
    let resolveManual: (update: Update | null) => void = () => {};
    checkMock
      .mockReturnValueOnce(new Promise((resolve) => (resolveSilent = resolve)))
      .mockReturnValueOnce(new Promise((resolve) => (resolveManual = resolve)));
    const { result } = renderHook(() => useUpdater());

    let silent: Promise<void> = Promise.resolve();
    let manual: Promise<void> = Promise.resolve();
    act(() => {
      silent = result.current.checkForUpdates({ silent: true });
      manual = result.current.checkForUpdates();
    });
    act(() => result.current.setDialogOpen(false));
    expect(result.current.state).toBe("checking");
    expect(result.current.dialogOpen).toBe(false);

    await act(async () => {
      resolveManual(null);
      await manual;
      resolveSilent(fakeUpdate());
      await silent;
    });

    expect(result.current.state).toBe("up-to-date");
    expect(result.current.release).toBeNull();
  });

  it("installs with progress and relaunches", async () => {
    const downloadAndInstall = vi.fn(async (onEvent: (event: DownloadEvent) => void) => {
      onEvent({ event: "Started", data: { contentLength: 200 } });
      onEvent({ event: "Progress", data: { chunkLength: 100 } });
    });
    checkMock.mockResolvedValue(fakeUpdate(downloadAndInstall));
    const { result } = renderHook(() => useUpdater());
    await act(() => result.current.checkForUpdates({ silent: true }));

    await act(() => result.current.installUpdate());

    expect(downloadAndInstall).toHaveBeenCalledOnce();
    expect(result.current.progress).toBe(50);
    expect(relaunch).toHaveBeenCalledOnce();
  });

  it("keeps the dialog open while installing", async () => {
    let finishInstall: () => void = () => {};
    const downloadAndInstall = vi.fn(() => new Promise<void>((resolve) => (finishInstall = resolve)));
    checkMock.mockResolvedValue(fakeUpdate(downloadAndInstall));
    const { result } = renderHook(() => useUpdater());
    await act(() => result.current.checkForUpdates({ silent: true }));

    let install: Promise<void> = Promise.resolve();
    act(() => {
      install = result.current.installUpdate();
    });
    act(() => result.current.setDialogOpen(false));

    expect(result.current.state).toBe("installing");
    expect(result.current.dialogOpen).toBe(true);
    await act(async () => {
      finishInstall();
      await install;
    });
  });

  it("reports a failed install", async () => {
    const downloadAndInstall = vi.fn().mockRejectedValue(new Error("bad signature"));
    checkMock.mockResolvedValue(fakeUpdate(downloadAndInstall));
    const { result } = renderHook(() => useUpdater());
    await act(() => result.current.checkForUpdates({ silent: true }));

    await act(() => result.current.installUpdate());

    expect(result.current.state).toBe("error");
    expect(result.current.error).toBe("bad signature");
    expect(relaunch).not.toHaveBeenCalled();
  });
});
