import { useEffect, useState } from "react";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { formats } from "@/lib/types";
import { useConverter } from "@/hooks/use-converter";
import { ConverterHeader } from "@/components/converter-header";
import { FormatButton } from "@/components/format-button";
import {
  FileJson,
  X,
  Upload,
  Check,
  Loader2,
  Clock,
  Download,
  FolderOpen,
} from "lucide-react";

export function ConverterScreen({ onBack }: { onBack: () => void }) {
  const {
    selectedFile,
    selectedFileName,
    selectedFormat,
    setSelectedFormat,
    isConverting,
    progress,
    result,
    error,
    elapsedSeconds,
    selectFile,
    removeFile,
    setFileFromPath,
    handleConvert,
    resetState,
    getProgressPercentage,
    formatElapsedTime,
    getDownloadRate,
  } = useConverter();

  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const webview = getCurrentWebview();
    const unlisten = webview.onDragDropEvent((event) => {
      if (event.payload.type === "over") {
        setIsDragging(true);
      } else if (event.payload.type === "leave") {
        setIsDragging(false);
      } else if (event.payload.type === "drop") {
        setIsDragging(false);
        const paths = event.payload.paths;
        if (paths.length > 0) {
          const file = paths[0];
          if (file.endsWith(".ndjson") || file.endsWith(".jsonl")) {
            setFileFromPath(file);
          }
        }
      }
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, [setFileFromPath]);

  // When no file selected, show centered upload card
  if (!selectedFile) {
    return (
      <div className="flex min-h-screen flex-col bg-background">
        <ConverterHeader onBack={onBack} />

        <main className="flex flex-1 items-center justify-center px-4">
          <Card
            className={`w-full max-w-md cursor-pointer border-2 border-dashed p-10 text-center transition-colors hover:border-primary/50 ${isDragging ? "border-primary bg-primary/5" : ""}`}
            onClick={selectFile}
          >
            <div className="flex flex-col items-center gap-4">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
                <Download className="h-8 w-8 text-primary" />
              </div>
              <div>
                <p className="text-lg font-medium">
                  {isDragging ? "Drop file here" : "Click or drag NDJSON file"}
                </p>
                <p className="text-sm text-muted-foreground">
                  Supports .ndjson and .jsonl files
                </p>
              </div>
              <Button className="mt-2">
                <Upload className="mr-2 h-4 w-4" />
                Choose File
              </Button>
            </div>
          </Card>
        </main>
      </div>
    );

  }

  return (
    <div className="min-h-screen bg-background">
      <ConverterHeader onBack={onBack} />

      <main className="container mx-auto px-4 py-8">
        {/* Main Content */}
        <div className="mx-auto max-w-2xl space-y-6">
          {/* File Upload */}
          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                  <FileJson className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <p className="font-medium">{selectedFileName}</p>
                  <p className="text-sm text-muted-foreground">
                    Ready to convert
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={removeFile}
                className="text-muted-foreground hover:text-destructive"
              >
                <X className="h-5 w-5" />
              </Button>
            </div>
          </Card>

          {/* Format Selector */}
          {selectedFile && (
            <div className="space-y-3">
              <h3 className="text-lg font-semibold">Popular Download Formats</h3>
              <div className="flex flex-wrap gap-2">
                {formats.map((format) => (
                  <FormatButton
                    key={format.name}
                    format={format}
                    isSelected={selectedFormat?.name === format.name}
                    disabled={!format.available || isConverting || !!result}
                    onClick={() => format.available && !isConverting && !result && setSelectedFormat(format)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Convert Button & Progress */}
          {selectedFormat && (
            <div className="space-y-4">
              {!result ? (
                <>
                  {/* Progress bar above convert button */}
                  {isConverting && progress && (
                    <Card className="space-y-4 p-4">
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="font-medium">
                            {progress.phase === "downloading"
                              ? "Downloading Images"
                              : progress.phase === "converting"
                                ? "Converting Annotations"
                                : progress.phase === "zipping"
                                  ? "Creating ZIP"
                                  : progress.phase === "parsing"
                                    ? "Parsing NDJSON"
                                    : "Processing"}
                          </span>
                          <span className="text-muted-foreground">
                            {progress.current} / {progress.total}
                          </span>
                        </div>
                        <Progress value={getProgressPercentage()} />
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <span className="flex-1 truncate">
                            {progress.item || "\u00A0"}
                          </span>
                          <div className="ml-2 flex shrink-0 items-center gap-3">
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {formatElapsedTime(elapsedSeconds)}
                            </span>
                            {progress.phase === "downloading" && (
                              <span>{getDownloadRate()} img/s</span>
                            )}
                          </div>
                        </div>
                      </div>
                    </Card>
                  )}

                  <Button
                    onClick={handleConvert}
                    disabled={isConverting}
                    className="w-full py-6 text-lg"
                    size="lg"
                  >
                    {isConverting ? (
                      <>
                        <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                        {progress?.phase === "downloading"
                          ? "Downloading Images..."
                          : progress?.phase === "converting"
                            ? "Converting..."
                            : progress?.phase === "zipping"
                              ? "Creating ZIP..."
                              : "Starting..."}
                      </>
                    ) : (
                      "Convert to " + selectedFormat.name
                    )}
                  </Button>
                </>
              ) : (
                <div className="space-y-4">
                  <Card className="bg-primary/5 p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/20">
                        <Check className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="font-semibold text-foreground">ZIP Downloaded</p>
                        <p className="text-sm text-muted-foreground">Your converted dataset is ready</p>
                      </div>
                    </div>
                    <div className="rounded-lg bg-background/50 p-3">
                      <p className="text-xs text-muted-foreground mb-1">Saved to:</p>
                      <p className="text-sm font-mono text-foreground break-all">
                        {result.zip_path}
                      </p>
                    </div>
                    {result.failed_downloads > 0 && (
                      <p className="mt-3 text-xs text-amber-700">
                        {result.failed_downloads === result.download_total
                          ? `All ${result.download_total} images failed to download. Check your network or CDN access.`
                          : `${result.failed_downloads} image${result.failed_downloads === 1 ? "" : "s"} failed to download and were omitted.`}
                      </p>
                    )}
                    <Button
                      onClick={() => revealItemInDir(result.zip_path)}
                      variant="outline"
                      className="w-full mt-4"
                    >
                      <FolderOpen className="mr-2 h-4 w-4" />
                      {navigator.platform.includes("Win") ? "Show in Explorer" : navigator.platform.includes("Mac") ? "Show in Finder" : "Show in Files"}
                    </Button>
                  </Card>
                  <Button
                    onClick={resetState}
                    className="w-full py-6 text-lg"
                    size="lg"
                  >
                    Convert Another File
                  </Button>
                </div>
              )}
            </div>
          )}

          {/* Error Display */}
          {error && (
            <Card className="bg-destructive/10 p-4 text-center">
              <p className="font-medium text-destructive">{error}</p>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
}
