(() => {
  const data = window.MG2RAG_DEMO;
  const svg = document.querySelector("#graph");
  const viewport = document.querySelector("#viewport");
  const edgeLayer = document.querySelector("#edges");
  const nodeLayer = document.querySelector("#nodes");
  const inspector = document.querySelector("#inspector");
  const shell = document.querySelector("#graph-shell");
  const NS = "http://www.w3.org/2000/svg";
  const nodes = new Map(data.nodes.map(node => [node.id, {...node}]));
  const enabledTypes = new Set(Object.keys(data.nodeTypes));
  let transform = {x: 0, y: 0, scale: 1};
  let drag = null;
  let pan = null;

  const make = (name, attrs = {}) => {
    const element = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  };

  const renderFilters = () => {
    const root = document.querySelector("#filters");
    Object.entries(data.nodeTypes).forEach(([type, meta]) => {
      const button = document.createElement("button");
      button.className = "filter active";
      button.style.setProperty("--node-color", meta.color);
      button.innerHTML = `<i></i><span>${meta.label}</span>`;
      button.addEventListener("click", () => {
        enabledTypes.has(type) ? enabledTypes.delete(type) : enabledTypes.add(type);
        button.classList.toggle("active", enabledTypes.has(type));
        updateVisibility();
      });
      root.appendChild(button);
    });
  };

  const edgeElements = data.edges.map(edge => {
    const group = make("g");
    const path = make("path", {class: `edge ${edge.type}`});
    const label = make("text", {class: "edge-label"});
    label.textContent = edge.label;
    group.append(path, label);
    edgeLayer.appendChild(group);
    return {edge, group, path, label};
  });

  const nodeElements = data.nodes.map(original => {
    const node = nodes.get(original.id);
    const meta = data.nodeTypes[node.type];
    const group = make("g", {class: `node${node.active ? " active" : ""}`, "data-id": node.id, "data-type": node.type});
    group.style.setProperty("--node-color", meta.color);
    group.append(
      make("circle", {class: "halo", r: 35}),
      make("circle", {class: "body", r: 30})
    );
    if (node.image) group.append(make("image", {href: node.image, x: -25, y: -25, width: 50, height: 50, preserveAspectRatio: "xMidYMid slice"}));
    const label = make("text", {class: "label", y: 48});
    label.textContent = node.label;
    const subtitle = make("text", {class: "subtitle", y: 63});
    subtitle.textContent = node.subtitle || "";
    group.append(label, subtitle);
    group.addEventListener("pointerdown", event => {
      event.stopPropagation();
      drag = {node, pointerId: event.pointerId};
      group.setPointerCapture(event.pointerId);
    });
    group.addEventListener("click", event => { event.stopPropagation(); inspect(node); });
    nodeLayer.appendChild(group);
    return {node, group};
  });

  const update = () => {
    nodeElements.forEach(({node, group}) => group.setAttribute("transform", `translate(${node.x} ${node.y})`));
    edgeElements.forEach(({edge, path, label}) => {
      const source = nodes.get(edge.source), target = nodes.get(edge.target);
      const dx = target.x - source.x, dy = target.y - source.y;
      const length = Math.max(Math.hypot(dx, dy), 1);
      const offset = 32;
      const x1 = source.x + dx / length * offset, y1 = source.y + dy / length * offset;
      const x2 = target.x - dx / length * offset, y2 = target.y - dy / length * offset;
      const curve = Math.min(45, length * .12);
      path.setAttribute("d", `M${x1},${y1} Q${(x1+x2)/2-dy/length*curve},${(y1+y2)/2+dx/length*curve} ${x2},${y2}`);
      label.setAttribute("x", (x1+x2)/2 - dy/length*curve*.5);
      label.setAttribute("y", (y1+y2)/2 + dx/length*curve*.5);
    });
  };

  const updateVisibility = () => {
    nodeElements.forEach(({node, group}) => group.classList.toggle("hidden", !enabledTypes.has(node.type)));
    edgeElements.forEach(({edge, group}) => {
      const visible = enabledTypes.has(nodes.get(edge.source).type) && enabledTypes.has(nodes.get(edge.target).type);
      group.classList.toggle("hidden", !visible);
    });
  };

  const inspect = node => {
    const meta = data.nodeTypes[node.type];
    inspector.style.setProperty("--node-color", meta.color);
    inspector.innerHTML = `
      <span class="inspector-tag">${meta.label}</span>
      <h3>${node.label}</h3><div class="sub">${node.subtitle || ""}</div>
      ${node.image ? `<img src="${node.image}" alt="${node.label}">` : ""}
      <p class="detail">${node.detail}</p>
      ${node.score !== undefined ? `<div class="score-meter" style="--score-width:${Math.min(node.score * 100,100)}%"><div><span>Activation / score</span><b>${node.score.toFixed(4)}</b></div><i></i></div>` : ""}`;
    nodeElements.forEach(item => item.group.classList.toggle("active", item.node.id === node.id || item.node.active));
  };

  const applyTransform = () => viewport.setAttribute("transform", `translate(${transform.x} ${transform.y}) scale(${transform.scale})`);
  const graphPoint = event => {
    const rect = svg.getBoundingClientRect();
    return {x: (event.clientX - rect.left - transform.x) / transform.scale, y: (event.clientY - rect.top - transform.y) / transform.scale};
  };

  svg.addEventListener("pointerdown", event => {
    if (event.target.closest?.(".node")) return;
    pan = {x: event.clientX, y: event.clientY, originX: transform.x, originY: transform.y, pointerId: event.pointerId};
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("pointermove", event => {
    if (drag) {
      const point = graphPoint(event); drag.node.x = point.x; drag.node.y = point.y; update();
    } else if (pan) {
      transform.x = pan.originX + event.clientX - pan.x; transform.y = pan.originY + event.clientY - pan.y; applyTransform();
    }
  });
  svg.addEventListener("pointerup", () => { drag = null; pan = null; });
  shell.addEventListener("wheel", event => {
    event.preventDefault();
    transform.scale = Math.max(.45, Math.min(2.2, transform.scale * Math.exp(-event.deltaY * .001)));
    applyTransform();
  }, {passive: false});
  document.querySelector("#reset-view").addEventListener("click", () => { transform = {x: 0, y: 0, scale: 1}; applyTransform(); });

  const maxScore = data.ranking[0].score;
  document.querySelector("#ranking").innerHTML = data.ranking.map((item, index) => `
    <div class="rank-row"><span class="rank">${String(index+1).padStart(2,"0")}</span><div><b>${item.title}</b> <small>${item.id}</small> ${item.gold ? '<span class="gold-pill">GOLD</span>' : ""}</div><div class="score-bar"><i style="width:${item.score/maxScore*100}%"></i></div><small>${item.score.toFixed(4)}</small></div>`).join("");

  renderFilters(); update(); inspect(nodes.get("chunk"));
})();
