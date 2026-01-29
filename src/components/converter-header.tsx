import { ArrowRight } from "lucide-react";
import { UpdateChecker } from "./update-checker";

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
        <div />
        <UpdateChecker />
      </div>
    </header>
  );

}
