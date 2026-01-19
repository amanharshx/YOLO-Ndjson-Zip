import { useState } from "react";
import { cn } from "@/lib/utils";
import type { Format } from "@/lib/types";

export function FormatButton({
  format,
  isSelected,
  disabled,
  onClick,
}: {
  format: Format;
  isSelected: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={onClick}
        disabled={disabled}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        className={cn(
          "rounded-lg border-2 px-4 py-2 text-sm font-medium transition-all",
          isSelected
            ? "border-primary bg-primary/10 text-primary"
            : format.available
              ? "border-border text-foreground hover:border-primary/50 hover:bg-primary/5"
              : "border-border/70 text-muted-foreground/70 cursor-not-allowed",
          disabled && !isSelected && "cursor-not-allowed opacity-50 hover:border-border hover:bg-transparent"
        )}
      >
        {format.name}
        {!format.available && <span className="ml-1 text-xs opacity-60">(Soon)</span>}
      </button>
      {showTooltip && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-50 pointer-events-none">
          <div className="bg-card border border-border rounded-lg px-3 py-2 shadow-lg text-xs w-max max-w-[320px] text-center whitespace-normal">
            <span className="text-primary">{format.desc}</span>
            <span className="text-muted-foreground"> used with </span>
            <span className="text-primary">{format.highlight}</span>
            <span className="text-muted-foreground">.</span>
          </div>
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-[1px]">
            <div className="border-6 border-transparent border-t-border" />
            <div className="absolute top-0 left-1/2 -translate-x-1/2 border-[5px] border-transparent border-t-card" />
          </div>
        </div>
      )}
    </div>
  );
}
