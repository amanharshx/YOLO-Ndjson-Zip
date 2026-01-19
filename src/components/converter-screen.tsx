import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ConverterHeader } from "@/components/converter-header";
import { Download, Upload } from "lucide-react";

export function ConverterScreen({ onBack }: { onBack: () => void }) {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <ConverterHeader onBack={onBack} />

      <main className="flex flex-1 items-center justify-center px-4">
        <Card
          className="w-full max-w-md cursor-pointer border-2 border-dashed p-10 text-center transition-colors hover:border-primary/50"
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
