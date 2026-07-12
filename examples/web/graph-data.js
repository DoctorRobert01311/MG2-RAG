window.MG2RAG_DEMO = {
  nodeTypes: {
    chunk: { label: "Chunk node", color: "#2697df" },
    sentence: { label: "Sentence index", color: "#77bff0" },
    entity: { label: "Unified multimodal node", color: "#68bd9e" },
    image: { label: "Image node", color: "#e9b13d" },
    object: { label: "Object index", color: "#e8ca78" }
  },
  nodes: [
    { id: "sent", type: "sentence", label: "Gestation lasts…", subtitle: "auxiliary sentence index", x: 160, y: 110, score: 0.86, detail: "Auxiliary dense index: sentence matches are aggregated onto linked unified multimodal nodes through M_SMI. This is not a direct PPR graph vertex." },
    { id: "obj", type: "object", label: "Impala region", subtitle: "auxiliary object index", x: 160, y: 450, image: "../showcase/impala/segmented_object.png", score: 0.6972, detail: "Auxiliary visual evidence preserved inside the unified node. Object matches are aggregated through M_OMI. Segmentation is an intermediate grounding mechanism, not the final graph output." },
    { id: "impala", type: "entity", label: "Impala", subtitle: "unified multimodal node", x: 420, y: 280, score: 0.92, active: true, detail: "A primary graph node aligning the textual entity with grounded visual evidence while preserving both modalities." },
    { id: "africa", type: "entity", label: "Africa", subtitle: "unified entity node", x: 500, y: 500, score: 0.28, detail: "A named entity connected through document context and extracted semantic structure." },
    { id: "iucn", type: "entity", label: "IUCN", subtitle: "unified entity node", x: 550, y: 85, score: 0.19, detail: "Entity participating in the semantic relation: International Union for Conservation of Nature is_also IUCN." },
    { id: "kbimg", type: "image", label: "Impala image", subtitle: "image node · Q132576", x: 700, y: 465, image: "../showcase/impala/kb_image.jpg", score: 0.71, detail: "Primary graph image node connected to its source chunk and grounded unified multimodal nodes." },
    { id: "chunk", type: "chunk", label: "Impala chunk", subtitle: "chunk node · Q132576", x: 790, y: 265, score: 0.1292, active: true, detail: "Primary graph chunk node connected to its associated image and unified multimodal nodes. It is ranked first after PPR." },
    { id: "wilddog", type: "chunk", label: "African wild dog", subtitle: "chunk node · Q173651", x: 1030, y: 110, score: 0.0370, detail: "A neighboring chunk in the constructed tiny multimodal knowledge graph." },
    { id: "elymus", type: "chunk", label: "Elymus hystrix", subtitle: "chunk node · Q5368484", x: 1050, y: 470, score: 0.0274, detail: "A distractor chunk connected to its own entities and source image in the full graph." }
  ],
  edges: [
    { source: "sent", target: "impala", type: "retrieval", label: "M_SMI aggregation" },
    { source: "sent", target: "iucn", type: "retrieval", label: "M_SMI aggregation" },
    { source: "obj", target: "impala", type: "retrieval", label: "M_OMI aggregation" },
    { source: "kbimg", target: "impala", type: "grounding", label: "grounding 0.6972" },
    { source: "chunk", target: "kbimg", type: "contextual", label: "source image" },
    { source: "chunk", target: "impala", type: "contextual", label: "mentions" },
    { source: "chunk", target: "africa", type: "contextual", label: "mentions" },
    { source: "impala", target: "africa", type: "semantic", label: "found in" },
    { source: "impala", target: "iucn", type: "semantic", label: "classified by" },
    { source: "impala", target: "wilddog", type: "retrieval", label: "graph propagation" },
    { source: "africa", target: "elymus", type: "retrieval", label: "graph propagation" }
  ],
  ranking: [
    { id: "Q132576", title: "Impala", score: 0.1291668713, gold: true },
    { id: "Q173651", title: "African wild dog", score: 0.0370077603 },
    { id: "Q5368484", title: "Elymus hystrix", score: 0.0274210740 },
    { id: "Q178185", title: "Heat engine", score: 0.0219943114 },
    { id: "Q1510231", title: "Betula nigra", score: 0.0172499716 }
  ]
};
