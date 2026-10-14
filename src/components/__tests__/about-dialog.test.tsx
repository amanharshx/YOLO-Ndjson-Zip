import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AboutDialog } from "../about-dialog";
import { UpdateDialog } from "../update-dialog";
import type { UpdaterController } from "@/hooks/use-updater";

vi.mock("@tauri-apps/plugin-opener", () => ({ openUrl: vi.fn() }));

function fakeUpdater(overrides: Partial<UpdaterController> = {}): UpdaterController {
  return {
    state: "idle",
    dialogOpen: true,
    release: null,
    progress: null,
    error: "",
    checkForUpdates: vi.fn(),
    installUpdate: vi.fn(),
    setDialogOpen: vi.fn(),
    ...overrides,
  };
}

describe("AboutDialog", () => {
  it("shows the installed version and hands off to the update check", () => {
    const updater = fakeUpdater();
    const onOpenChange = vi.fn();
    render(<AboutDialog open updater={updater} onOpenChange={onOpenChange} />);

    expect(screen.getByRole("button", { name: __APP_VERSION__ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Check for Updates/ }));

    expect(onOpenChange).toHaveBeenCalledWith(false);
    expect(updater.checkForUpdates).toHaveBeenCalledOnce();
  });
});

describe("UpdateDialog", () => {
  it("offers install with the release details when an update is available", () => {
    const updater = fakeUpdater({
      state: "available",
      release: { version: "9.9.9", date: "", notes: "Faster exports" },
    });
    render(<UpdateDialog updater={updater} />);

    expect(screen.getByText("Faster exports")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Install and restart" }));

    expect(updater.installUpdate).toHaveBeenCalledOnce();
  });

  it("can be closed while checking", () => {
    render(<UpdateDialog updater={fakeUpdater({ state: "checking" })} />);

    expect(screen.getByRole("button", { name: "Close" })).toBeInTheDocument();
  });

  it("hides the close button while installing", () => {
    render(<UpdateDialog updater={fakeUpdater({ state: "installing", progress: 40 })} />);

    expect(screen.getByText("Downloading… 40%")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Close" })).not.toBeInTheDocument();
  });
});
