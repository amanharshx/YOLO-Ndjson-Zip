import { useEffect, useRef, useState } from "react";
import { check } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";
import { RefreshCw, Download, CheckCircle, AlertCircle } from "lucide-react";

type UpdateState = "idle" | "checking" | "available" | "downloading" | "ready" | "up-to-date" | "error";

export function UpdateChecker() {
  const [state, setState] = useState<UpdateState>("idle");
  const [version, setVersion] = useState<string>("");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string>("");
  const resetTimeoutRef = useRef<number | null>(null);

  const clearResetTimeout = () => {
    if (resetTimeoutRef.current !== null) {
      window.clearTimeout(resetTimeoutRef.current);
      resetTimeoutRef.current = null;
    }
  };

  const scheduleResetToIdle = (delayMs: number) => {
    clearResetTimeout();
    resetTimeoutRef.current = window.setTimeout(() => {
      setState("idle");
      resetTimeoutRef.current = null;
    }, delayMs);
  };

  useEffect(() => {
    return () => {
      clearResetTimeout();
    };
  }, []);

  const checkForUpdates = async () => {
    setState("checking");
    setError("");

    try {
      const update = await check();

      if (update) {
        setVersion(update.version);
        setState("available");
      } else {
        setState("up-to-date");
        scheduleResetToIdle(3000);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to check for updates");
      setState("error");
      scheduleResetToIdle(5000);
    }
  };

  const downloadAndInstall = async () => {
    setState("downloading");

    try {
      const update = await check();
      if (!update) {
        setState("idle");
        return;
      }

      let downloaded = 0;
      let contentLength = 0;

      await update.downloadAndInstall((event) => {
        if (event.event === "Started") {
          contentLength = event.data.contentLength ?? 0;
        } else if (event.event === "Progress") {
          downloaded += event.data.chunkLength;
          if (contentLength > 0) {
            setProgress(Math.round((downloaded / contentLength) * 100));
          }
        }
      });

      setState("ready");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to download update");
      setState("error");
    }
  };

  const restartApp = async () => {
    await relaunch();
  };

  if (state === "idle") {
    return (
      <button
        onClick={checkForUpdates}
        className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
        title="Check for updates"
      >
        <RefreshCw className="h-3.5 w-3.5" />
        Updates
      </button>
    );
  }

  if (state === "checking") {
    return (
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <RefreshCw className="h-3.5 w-3.5 animate-spin" />
        Checking...
      </span>
    );
  }

  if (state === "up-to-date") {
    return (
      <span data-testid="update-status" className="flex items-center gap-1.5 text-xs text-green-600">
        <CheckCircle className="h-3.5 w-3.5" />
        Up to date
      </span>
    );
  }

  if (state === "available") {
    return (
      <button
        onClick={downloadAndInstall}
        className="flex items-center gap-1.5 text-xs text-blue-600 hover:text-blue-700 font-medium"
      >
        <Download className="h-3.5 w-3.5" />
        Update to {version}
      </button>
    );
  }

  if (state === "downloading") {
    return (
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <RefreshCw className="h-3.5 w-3.5 animate-spin" />
        Downloading... {progress}%
      </span>
    );
  }

  if (state === "ready") {
    return (
      <button
        onClick={restartApp}
        className="flex items-center gap-1.5 text-xs text-green-600 hover:text-green-700 font-medium"
      >
        <RefreshCw className="h-3.5 w-3.5" />
        Restart to update
      </button>
    );
  }

  if (state === "error") {
    return (
      <span className="flex items-center gap-1.5 text-xs text-red-600" title={error}>
        <AlertCircle className="h-3.5 w-3.5" />
        Update failed
      </span>
    );
  }

  return null;
}
