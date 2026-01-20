import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { formats } from "@/lib/types";
import { useConverter } from "@/hooks/use-converter";
import { ConverterHeader } from "@/components/converter-header";
import { FormatButton } from "@/components/format-button";
import { FileJson, X, Upload, Download } from "lucide-react";

export function ConverterScreen({ onBack }: { onBack: () => void }) {
  const {
    selectedFile,
    selectedFileName,
    selectedFormat,
    setSelectedFormat,
    selectFile,
    removeFile,
  } = useConverter();

  if (!selectedFile) {
    return (
      <div className="flex min-h-screen flex-col bg-background">
        <ConverterHeader onBack={onBack} />

        <main className="flex flex-1 items-center justify-center px-4">
          <Card
            className="w-full max-w-md cursor-pointer border-2 border-dashed p-10 text-center transition-colors hover:border-primary/50"
            onClick={selectFile}
          >
            <div className="flex flex-col items-center gap-4">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
                <Download className="h-8 w-8 text-primary" />
              </div>
              <div>
                <p className="text-lg font-medium">
                  Click to select NDJSON file
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
          <div className="space-y-3">
            <h3 className="text-lg font-semibold">Popular Download Formats</h3>
            <div className="flex flex-wrap gap-2">
              {formats.map((format) => (
                <FormatButton
                  key={format.name}
                  format={format}
                  isSelected={selectedFormat?.name === format.name}
                  disabled={!format.available}
                  onClick={() => format.available && setSelectedFormat(format)}
                />
              ))}
            </div>
          </div>

          {/* Convert Button */}
          {selectedFormat && (
            <Button className="w-full py-6 text-lg" size="lg">
              Convert to {selectedFormat.name}
            </Button>
          )}
        </div>
      </main>
    </div>
  );
}
