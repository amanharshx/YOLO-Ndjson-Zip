import { openUrl } from "@tauri-apps/plugin-opener";
import { ExternalLink, Github, Info, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { UpdaterController } from "@/hooks/use-updater";
import appIcon from "../../assets/icon2.png";

const repository = "github.com/amanharshx/yolo-ndjson-zip";
const repositoryUrl = `https://${repository}`;

export interface AboutProps {
  onOpenAbout: () => void;
  updateAvailable: boolean;
}

export function AboutButton({ onOpenAbout, updateAvailable }: AboutProps) {
  return (
    <button
      type="button"
      onClick={onOpenAbout}
      className="relative flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
      title="About YOLO NDJSON Converter"
    >
      <Info className="h-3.5 w-3.5" />
      About
      {updateAvailable && (
        <span
          className="absolute -top-1 -right-1.5 h-2 w-2 rounded-full bg-blue-500 ring-2 ring-background"
          aria-label="Update available"
        />
      )}
    </button>
  );
}

export function AboutDialog({
  open,
  updater,
  onOpenChange,
}: {
  open: boolean;
  updater: UpdaterController;
  onOpenChange: (open: boolean) => void;
}) {
  const checkForUpdates = () => {
    onOpenChange(false);
    void updater.checkForUpdates();
  };
  const openRepository = () => {
    openUrl(repositoryUrl).catch((e) => console.error("Failed to open repository:", e));
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>About</DialogTitle>
          <DialogDescription>About YOLO NDJSON Converter and application updates.</DialogDescription>
        </DialogHeader>

        <div className="flex flex-col items-center py-3 text-center">
          <img src={appIcon} alt="" className="h-14 w-14 drop-shadow-sm" />
          <div className="mt-3 flex items-center gap-2">
            <h2 className="text-xl font-semibold">YOLO NDJSON to ZIP</h2>
            <button
              type="button"
              className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                updater.state === "available"
                  ? "bg-amber-50 text-amber-700"
                  : "bg-green-50 text-green-600"
              }`}
              title="Check for updates"
              onClick={checkForUpdates}
            >
              {__APP_VERSION__}
            </button>
          </div>
          <p className="mt-1 max-w-md text-sm text-muted-foreground">
            Convert your NDJSON annotation exports to popular ML formats. Fast, private, and runs
            entirely on your machine.
          </p>
        </div>

        <div className="flex items-center gap-3 rounded-xl border p-4">
          <Github className="h-5 w-5 shrink-0" />
          <div className="min-w-0 flex-1">
            <p className="font-medium">Open source</p>
            <p className="truncate text-sm text-muted-foreground">MIT · {repository}</p>
          </div>
          <Button variant="ghost" size="sm" onClick={openRepository}>
            View repository
            <ExternalLink className="ml-1.5 h-3.5 w-3.5" />
          </Button>
        </div>

        <div className="flex justify-end">
          <Button variant="outline" size="sm" onClick={checkForUpdates}>
            <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
            Check for Updates
          </Button>
        </div>

        <DialogFooter>
          <Button onClick={() => onOpenChange(false)}>Done</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
