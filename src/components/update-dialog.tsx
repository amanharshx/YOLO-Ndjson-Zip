import type { ReactNode } from "react";
import { Loader2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { formatReleaseDate, type UpdaterController } from "@/hooks/use-updater";

function ReleaseDetails({ version, date, notes }: { version: string; date: string; notes: string }) {
  const formattedDate = formatReleaseDate(date);
  return (
    <div className="space-y-3">
      <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-sm">
        <dt className="text-muted-foreground">Version</dt>
        <dd className="truncate font-mono">{version}</dd>
        {formattedDate && (
          <>
            <dt className="text-muted-foreground">Released</dt>
            <dd className="truncate">{formattedDate}</dd>
          </>
        )}
      </dl>
      {notes && (
        <section className="max-h-72 overflow-y-auto border-t pt-3 text-sm whitespace-pre-wrap">
          {notes}
        </section>
      )}
    </div>
  );
}

export function UpdateDialog({ updater }: { updater: UpdaterController }) {
  const { state, setDialogOpen, release, progress, error } = updater;
  const blocking = state === "installing";
  const preventWhileBlocking = (event: Event) => {
    if (blocking) event.preventDefault();
  };

  // One branch per state; each owns its description, body, and footer so a
  // state can never show another state's copy.
  let description = "";
  let body: ReactNode = null;
  let footer: ReactNode = null;
  switch (state) {
    case "checking":
      description = "Checking GitHub for the latest release…";
      body = (
        <div className="flex justify-center py-6">
          <RefreshCw
            className="h-5 w-5 animate-spin text-muted-foreground"
            aria-label="Checking for updates"
          />
        </div>
      );
      break;
    case "available":
      description = `YOLO NDJSON Converter ${release?.version ?? ""} is ready to install. The app will restart automatically.`;
      body = release && <ReleaseDetails {...release} />;
      footer = (
        <DialogFooter>
          <Button variant="outline" onClick={() => setDialogOpen(false)}>
            Not now
          </Button>
          <Button onClick={() => void updater.installUpdate()}>Install and restart</Button>
        </DialogFooter>
      );
      break;
    case "installing":
      description = "Downloading and installing the update…";
      body =
        progress !== null ? (
          <div className="space-y-2 py-2">
            <Progress value={progress} aria-label={`Downloading update ${progress}%`} />
            <p className="text-sm text-muted-foreground tabular-nums" aria-live="polite">
              Downloading… {progress}%
            </p>
          </div>
        ) : (
          <div className="flex items-center justify-center gap-2 py-4">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" aria-hidden="true" />
            <span className="text-sm text-muted-foreground">Downloading…</span>
          </div>
        );
      break;
    case "up-to-date":
      description = "You have the latest version of YOLO NDJSON Converter.";
      body = <ReleaseDetails version={__APP_VERSION__} date="" notes="" />;
      footer = (
        <DialogFooter>
          <Button onClick={() => setDialogOpen(false)}>Done</Button>
        </DialogFooter>
      );
      break;
    case "error":
      description = `Update failed: ${error}`;
      footer = (
        <DialogFooter>
          <Button onClick={() => void updater.checkForUpdates()}>Try again</Button>
        </DialogFooter>
      );
      break;
    case "idle":
      break;
  }

  return (
    <Dialog open={updater.dialogOpen} onOpenChange={setDialogOpen}>
      <DialogContent
        className="sm:max-w-2xl"
        showCloseButton={!blocking}
        onEscapeKeyDown={preventWhileBlocking}
        onInteractOutside={preventWhileBlocking}
      >
        <DialogHeader>
          <DialogTitle>YOLO NDJSON Converter updates</DialogTitle>
          <DialogDescription aria-live="polite">{description}</DialogDescription>
        </DialogHeader>
        {body}
        {footer}
      </DialogContent>
    </Dialog>
  );
}
