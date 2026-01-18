import { useState } from "react";
import { WelcomeScreen } from "@/components/welcome-screen";

function App() {
  const [showConverter, setShowConverter] = useState(false);

  if (showConverter) {
    return <div>Converter coming soon...</div>;
  }

  return <WelcomeScreen onGetStarted={() => setShowConverter(true)} />;
}

export default App;
