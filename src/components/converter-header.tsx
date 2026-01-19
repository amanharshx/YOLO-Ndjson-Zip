import { ArrowRight } from "lucide-react";

export function ConverterHeader({ onBack }: { onBack: () => void }) {
  return (
    <header className="border-b bg-card">
      <div className="container mx-auto flex items-center justify-between px-4 py-3">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowRight className="h-4 w-4 rotate-180" />
          Back
        </button>
        <h1 className="text-sm font-semibold">YOLO NDJSON Converter</h1>
        <div className="w-16" />
      </div>
    </header>
  );
}
