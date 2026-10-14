import { useCallback, useRef, useState } from "react";
import { check, type Update } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";

// Applies to the check request only; the download keeps the plugin default.
const CHECK_TIMEOUT_MS = 10_000;

export type UpdateState = "idle" | "checking" | "available" | "installing" | "up-to-date" | "error";

export interface Release {
  version: string;
  date: string;
  notes: string;
}

export interface UpdaterController {
  state: UpdateState;
  dialogOpen: boolean;
  release: Release | null;
  progress: number | null;
  error: string;
  checkForUpdates: (opts?: { silent?: boolean }) => Promise<void>;
  installUpdate: () => Promise<void>;
  setDialogOpen: (open: boolean) => void;
}

function errorMessage(value: unknown, fallback: string): string {
  if (typeof value === "string" && value) return value;
  if (value instanceof Error && value.message) return value.message;
  return fallback;
}

export function formatReleaseDate(raw: string): string | null {
  if (!raw) return null;
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

export function useUpdater(): UpdaterController {
  const [state, setState] = useState<UpdateState>("idle");
  const [dialogOpen, setDialogOpenState] = useState(false);
  const [release, setRelease] = useState<Release | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState("");
  const updateRef = useRef<Update | null>(null);
  const stateRef = useRef<UpdateState>("idle");
  // Monotonic id so a stale check can never overwrite a newer one.
  const requestRef = useRef(0);

  const syncState = useCallback((next: UpdateState) => {
    stateRef.current = next;
    setState(next);
  }, []);

  const isBusy = () => stateRef.current === "checking" || stateRef.current === "installing";

  // Only an install locks the dialog; a check is safe to walk away from.
  const setDialogOpen = useCallback((open: boolean) => {
    if (!open && stateRef.current === "installing") return;
    setDialogOpenState(open);
  }, []);

  const rememberUpdate = useCallback((update: Update | null) => {
    updateRef.current = update;
    setRelease(
      update ? { version: update.version, date: update.date ?? "", notes: update.body ?? "" } : null,
    );
  }, []);

  // A silent check only marks an available update; it never opens the dialog
  // or reports "up to date".
  const checkForUpdates = useCallback(
    async ({ silent = false }: { silent?: boolean } = {}) => {
      if (isBusy()) {
        if (!silent) setDialogOpenState(true);
        return;
      }
      const requestId = ++requestRef.current;
      if (!silent) {
        setDialogOpenState(true);
        setError("");
        syncState("checking");
      }

      try {
        const update = await check({ timeout: CHECK_TIMEOUT_MS });
        if (requestId !== requestRef.current) return;
        rememberUpdate(update);
        syncState(update ? "available" : silent ? "idle" : "up-to-date");
      } catch (e) {
        if (requestId !== requestRef.current) return;
        rememberUpdate(null);
        if (silent) {
          console.warn("Background update check failed:", e);
          syncState("idle");
          return;
        }
        setError(errorMessage(e, "Failed to check for updates"));
        syncState("error");
      }
    },
    [rememberUpdate, syncState],
  );

  const installUpdate = useCallback(async () => {
    const update = updateRef.current;
    if (!update || stateRef.current !== "available") return;

    // Invalidate any in-flight check so it can never overwrite the install.
    ++requestRef.current;
    setDialogOpenState(true);
    setError("");
    setProgress(null);
    syncState("installing");

    let downloaded = 0;
    let total = 0;
    try {
      await update.downloadAndInstall((event) => {
        if (event.event === "Started") {
          total = event.data.contentLength ?? 0;
        } else if (event.event === "Progress") {
          downloaded += event.data.chunkLength;
          if (total > 0) setProgress(Math.min(100, Math.round((downloaded / total) * 100)));
        }
      });
      await relaunch();
    } catch (e) {
      setError(errorMessage(e, "Failed to install the update"));
      syncState("error");
    }
  }, [syncState]);

  return {
    state,
    dialogOpen,
    release,
    progress,
    error,
    checkForUpdates,
    installUpdate,
    setDialogOpen,
  };
}
