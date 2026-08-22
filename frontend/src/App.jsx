import { useState } from "react";
import Sidebar from "./Sidebar";
import ContainerPanel from "./ContainerPanel";
import DocumentPanel from "./DocumentPanel";
import "./App.css";

function App() {
  const [section, setSection] = useState("container");

  return (
    <div className="layout">
      <Sidebar active={section} onSelect={setSection} />
      <div className="content">
        {section === "container" && <ContainerPanel />}
        {section === "documento" && <DocumentPanel />}
      </div>
    </div>
  );
}

export default App;
