import { useState } from "react";
import { WelcomeScreen } from "@/components/welcome-screen";
import { ConverterScreen } from "@/components/converter-screen";

function App() {
  const  [showConverter, setShowConverter] = useState(false);

  if (showConverter) {
    return <ConverterScreen onBack={() => setShowConverter(false)} />;
  }

  return <WelcomeScreen onGetStarted={() => setShowConverter(true)} />;
}

export default App;
