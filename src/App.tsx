import { useEffect, useState } from "react";
import { WelcomeScreen } from "@/components/welcome-screen";
import { ConverterScreen } from "@/components/converter-screen";
import { AboutDialog } from "@/components/about-dialog";
import { UpdateDialog } from "@/components/update-dialog";
import { useUpdater } from "@/hooks/use-updater";

function App() {
  const [showConverter, setShowConverter] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const updater = useUpdater();
  const { checkForUpdates } = updater;

  // One background check per launch marks the About button when an update exists.
  useEffect(() => {
    if (!import.meta.env.DEV) void checkForUpdates({ silent: true });
  }, [checkForUpdates]);

  const about = {
    onOpenAbout: () => setAboutOpen(true),
    updateAvailable: updater.state === "available",
  };

  return (
    <>
      {showConverter ? (
        <ConverterScreen onBack={() => setShowConverter(false)} {...about} />
      ) : (
        <WelcomeScreen onGetStarted={() => setShowConverter(true)} {...about} />
      )}
      <AboutDialog open={aboutOpen} updater={updater} onOpenChange={setAboutOpen} />
      <UpdateDialog updater={updater} />
    </>
  );
}

export default App;
