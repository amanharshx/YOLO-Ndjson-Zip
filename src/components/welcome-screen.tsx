import { Button } from "@/components/ui/button";
import { formatBadges } from "@/lib/types";
import { Zap, Layers, Lock, ArrowRight } from "lucide-react";
import { UpdateChecker } from "./update-checker";
import appIcon from "../../assets/icon2.png";

export function WelcomeScreen({ onGetStarted }: { onGetStarted: () => void }) {
  return (
    <div data-testid="welcome-screen" className="relative flex min-h-screen">
      <div className="absolute top-4 right-4">
        <UpdateChecker />
      </div>
      {/* Left Side - Branding */}
      <div className="flex flex-1 flex-col items-center justify-center bg-primary/5 px-12 py-8">
        <div className="flex flex-col items-center max-w-lg">
          <img src={appIcon} alt="YOLO NDJSON Converter" className="mb-8 h-24 w-24" />
          <h1 className="mb-4 text-4xl font-bold tracking-tight text-foreground text-center">
            YOLO NDJSON
            <br />
            <span className="text-primary">to ZIP</span>
          </h1>
          <p className="mb-8 text-lg text-muted-foreground text-center">
            Convert your NDJSON annotation exports to popular ML formats.
            Fast, private, and runs entirely on your machine.
          </p>

          {/* Features list */}
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                <Zap className="h-4 w-4 text-primary" />
              </div>
              <span className="text-sm text-muted-foreground">
                <span className="font-semibold text-foreground">Fast</span> parallel image downloads
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                <Layers className="h-4 w-4 text-primary" />
              </div>
              <span className="text-sm text-muted-foreground">
                <span className="font-semibold text-foreground">12</span> formats across <span className="font-semibold text-foreground">4</span> task types
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                <Lock className="h-4 w-4 text-primary" />
              </div>
              <span className="text-sm text-muted-foreground">
                <span className="font-semibold text-foreground">100%</span> private - nothing leaves your machine
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Side - Get Started */}
      <div className="flex flex-1 flex-col justify-center px-12 py-8">
        <div className="max-w-md">
          <div className="space-y-6">
            <div>
              <h3 className="mb-3 text-sm font-medium text-muted-foreground">
                SUPPORTED FORMATS
              </h3>
              <div className="flex flex-wrap gap-2">
                {formatBadges.map((format) => (
                  <span
                    key={format}
                    className="rounded-full bg-black/[0.04] px-3 py-1 text-xs font-medium text-secondary-foreground"
                  >
                    {format}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <h3 className="mb-3 text-sm font-medium text-muted-foreground">
                TASK TYPES
              </h3>
              <div className="flex flex-wrap gap-2">
                {["Detection", "Segmentation", "Pose", "Classification"].map(
                  (task) => (
                    <span
                      key={task}
                      className="rounded-full border border-border px-3 py-1 text-xs font-medium text-muted-foreground"
                    >
                      {task}
                    </span>
                  )
                )}
              </div>
            </div>

            <Button
              data-testid="get-started"
              onClick={onGetStarted}
              size="lg"
              className="mt-4 w-full py-6 text-lg"
            >
              Get Started
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
