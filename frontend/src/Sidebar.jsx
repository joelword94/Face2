import "./Sidebar.css";

function Sidebar({ active, onSelect }) {
  const items = [
    { id: "container", label: "Container" },
    { id: "documento", label: "Documento" },
  ];

  return (
    <div className="sidebar">
      <h2 className="sidebar-title">Menú</h2>
      <ul>
        {items.map((item) => (
          <li
            key={item.id}
            className={active === item.id ? "active" : ""}
            onClick={() => onSelect(item.id)}
          >
            {item.label}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Sidebar;
