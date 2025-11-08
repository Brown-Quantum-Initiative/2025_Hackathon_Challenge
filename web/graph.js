const GRAPH_PATH = "../parser/graph6.json";
const DEFAULT_TIME_STEP = 0;
const ANIMATION_INTERVAL_MS = 500;

let cachedGraphData = null;
let availableTimeSteps = [];
let currentTimeStep = null;
let currentTransform = d3.zoomIdentity;
let resizeFrame = null;
let animationTimerId = null;
let nodeTooltipElement = null;

const NODE_TYPE_COLORS = Object.freeze({
  qubit: "#2563eb",
  classical_bit: "#0ea5e9",
  single_qubit_gate: "#10b981",
  one_quit_gate: "#8b5cf6",
  two_qubit_gate: "#f97316",
  measurement: "#dc2626",
  default: "#6b7280"
});

const NODE_TYPE_LABELS = Object.freeze({
  qubit: "Qubit",
  classical_bit: "Classical bit",
  single_qubit_gate: "Single-qubit gate",
  one_quit_gate: "Continuous Single-qubit gate",
  two_qubit_gate: "Two-qubit gate",
  measurement: "Measurement",
  default: "Other"
});

const NODE_TYPE_ORDER = Object.freeze([
  "qubit",
  "classical_bit",
  "single_qubit_gate",
  "one_quit_gate",
  "two_qubit_gate",
  "measurement",
  "default"
]);

const FALLBACK_COLOR_SCALE =
  typeof d3 !== "undefined" && typeof d3.scaleOrdinal === "function"
    ? d3.scaleOrdinal(d3.schemeCategory10)
    : null;

function normalizeNodeType(type) {
  return type ?? "default";
}

function getNodeColor(type) {
  const normalizedType = normalizeNodeType(type);
  if (Object.prototype.hasOwnProperty.call(NODE_TYPE_COLORS, normalizedType)) {
    return NODE_TYPE_COLORS[normalizedType];
  }
  if (FALLBACK_COLOR_SCALE) {
    return FALLBACK_COLOR_SCALE(normalizedType);
  }
  return NODE_TYPE_COLORS.default;
}

function getLegendLabel(type) {
  const normalizedType = normalizeNodeType(type);
  if (Object.prototype.hasOwnProperty.call(NODE_TYPE_LABELS, normalizedType)) {
    return NODE_TYPE_LABELS[normalizedType];
  }
  const readableLabel = normalizedType.replace(/[_-]+/g, " ");
  return readableLabel.charAt(0).toUpperCase() + readableLabel.slice(1);
}

function getLegendSortIndex(type) {
  const normalizedType = normalizeNodeType(type);
  const orderIndex = NODE_TYPE_ORDER.indexOf(normalizedType);
  return orderIndex >= 0 ? orderIndex : NODE_TYPE_ORDER.length;
}

function updateLegend(nodes) {
  const legend = document.getElementById("graph-legend");
  if (!legend) {
    return;
  }

  const list = legend.querySelector(".graph-legend__list");
  if (!list) {
    return;
  }

  const uniqueTypes = Array.from(
    new Set(nodes.map(node => normalizeNodeType(node.type)))
  ).sort((a, b) => {
    const orderDelta = getLegendSortIndex(a) - getLegendSortIndex(b);
    if (orderDelta !== 0) {
      return orderDelta;
    }
    return getLegendLabel(a).localeCompare(getLegendLabel(b));
  });

  list.innerHTML = "";

  if (!uniqueTypes.length) {
    legend.setAttribute("hidden", "");
    return;
  }

  legend.removeAttribute("hidden");
  uniqueTypes.forEach(typeKey => {
    const item = document.createElement("li");
    item.className = "graph-legend__item";

    const swatch = document.createElement("span");
    swatch.className = "graph-legend__swatch";
    swatch.style.backgroundColor = getNodeColor(typeKey);

    const label = document.createElement("span");
    label.className = "graph-legend__label";
    label.textContent = getLegendLabel(typeKey);

    item.appendChild(swatch);
    item.appendChild(label);
    list.appendChild(item);
  });
}

function getBitInfo(bit) {
  const match = bit.match(/([a-zA-Z]+)\[(\d+)\]/);
  if (match) {
    return [match[1], match[2]];
  }
  return [null, null];
}

function cloneGraph(graph) {
  return {
    nodes: graph.nodes.map(node => ({ ...node })),
    edges: graph.edges.map(edge => ({ ...edge }))
  };
}

function ensureNodeTooltipElement() {
  if (nodeTooltipElement && nodeTooltipElement.isConnected) {
    return nodeTooltipElement;
  }

  nodeTooltipElement = document.getElementById("node-tooltip");
  if (nodeTooltipElement && nodeTooltipElement.isConnected) {
    return nodeTooltipElement;
  }

  const wrapper = document.querySelector(".graph-wrapper");
  if (!wrapper) {
    return null;
  }

  nodeTooltipElement = document.createElement("div");
  nodeTooltipElement.id = "node-tooltip";
  nodeTooltipElement.className = "node-tooltip";
  nodeTooltipElement.setAttribute("hidden", "");
  wrapper.appendChild(nodeTooltipElement);
  return nodeTooltipElement;
}

function positionNodeTooltip(event) {
  const tooltip = ensureNodeTooltipElement();
  if (!tooltip) {
    return;
  }

  const wrapper = document.querySelector(".graph-wrapper");
  if (!wrapper) {
    return;
  }

  const wrapperRect = wrapper.getBoundingClientRect();
  const pointerX = event.clientX - wrapperRect.left;
  const pointerY = event.clientY - wrapperRect.top;

  tooltip.style.left = `${pointerX}px`;
  tooltip.style.top = `${pointerY}px`;
}

function showNodeTooltip(event, nodeDatum) {
  const tooltip = ensureNodeTooltipElement();
  if (!tooltip) {
    return;
  }

  tooltip.replaceChildren();

  const titleEl = document.createElement("div");
  titleEl.className = "node-tooltip__title";
  titleEl.textContent = nodeDatum.name ?? nodeDatum.id ?? "Continuous gate";
  tooltip.appendChild(titleEl);

  if (nodeDatum.gate_info) {
    const infoEl = document.createElement("div");
    infoEl.className = "node-tooltip__info";
    infoEl.textContent = nodeDatum.gate_info;
    tooltip.appendChild(infoEl);
  }

  tooltip.classList.add("is-visible");
  tooltip.removeAttribute("hidden");
  positionNodeTooltip(event);
}

function hideNodeTooltip() {
  const tooltip = ensureNodeTooltipElement();
  if (!tooltip) {
    return;
  }

  tooltip.classList.remove("is-visible");
  tooltip.setAttribute("hidden", "");
}

class QasmParser {
  constructor(qasmText) {
    this.qasmText = qasmText ?? "";
    this.lines = this._getSanitizedLines(this.qasmText);
    this.bits = {};
    this.gateIdCounter = 0;
    this.timestamps = {};
  }

  parse() {
    this._extractBits();
    this._buildGraph();
    return this.timestamps;
  }

  _getSanitizedLines(text) {
    return text
      .split(/\r?\n/)
      .map(line => line.split("//")[0].trim())
      .filter(Boolean);
  }

  _extractBits() {
    for (const line of this.lines) {
      if (line.startsWith("qreg")) {
        const match = line.match(/qreg\s+([a-zA-Z_]\w*)\[(\d+)\]/);
        if (match) {
          const [, registerName, sizeString] = match;
          const size = Number(sizeString);
          for (let i = 0; i < size; i += 1) {
            const bitId = `${registerName}${i}`;
            this.bits[bitId] = {
              id: bitId,
              type: "qubit",
              name: bitId,
              lastGateConnected: null
            };
          }
        }
      } else if (line.startsWith("creg")) {
        const match = line.match(/creg\s+([a-zA-Z_]\w*)\[(\d+)\]/);
        if (match) {
          const [, registerName, sizeString] = match;
          const size = Number(sizeString);
          for (let i = 0; i < size; i += 1) {
            const bitId = `${registerName}${i}`;
            this.bits[bitId] = {
              id: bitId,
              type: "classical_bit",
              name: bitId,
              lastGateConnected: null
            };
          }
        }
      }
    }
  }

  _buildGraph() {
    const currentGraph = {
      nodes: Object.values(this.bits).map(({ id, type, name }) => ({ id, type, name })),
      edges: []
    };

    this.timestamps[0] = cloneGraph(currentGraph);
    let lineCounter = 0;

    for (const line of this.lines) {
      if (
        line.startsWith("OPENQASM") ||
        line.startsWith("include") ||
        line.startsWith("qreg") ||
        line.startsWith("creg")
      ) {
        continue;
      }

      lineCounter += 1;
      let graphUpdated = false;

      if (line.includes("(") && line.includes(")")) {
        const parts = line.split(/[()]/);

        if (parts.length === 3) {
          graphUpdated = this._handleSingleQubitParamGate(parts, currentGraph);
        } else if (parts.length === 4) {
          graphUpdated = this._handleTwoQubitParamGate(parts, currentGraph, line);
        }
      } else {
        const parts = line
          .split(/[,\s]+/)
          .map(part => part.trim())
          .filter(Boolean);

        if (parts.length === 2) {
          graphUpdated = this._handleSingleQubitGate(parts, currentGraph);
        } else if (parts.length === 3) {
          graphUpdated = this._handleTwoQubitGate(parts, currentGraph, line);
        } else if (parts.length === 4 && parts[0] === "measure") {
          graphUpdated = this._handleMeasureGate(parts, currentGraph);
        }
      }

      if (graphUpdated) {
        this.timestamps[lineCounter] = cloneGraph(currentGraph);
      }
    }
  }

  _registerGateOnBit(bitId, gateId) {
    const bit = this.bits[bitId];
    if (!bit) {
      throw new Error(`Unknown bit referenced in gate: '${bitId}'`);
    }

    const source = bit.lastGateConnected ?? bitId;
    bit.lastGateConnected = gateId;
    return source;
  }

  _addGateNode(currentGraph, gateDescriptor) {
    currentGraph.nodes.push(gateDescriptor);
  }

  _handleSingleQubitParamGate(parts, currentGraph) {
    const gateName = parts[0].trim();
    const gateInfo = parts[1].trim();
    const bitToken = parts[2].trim().replace(/;$/, "");
    const [register, index] = getBitInfo(bitToken);
    if (!register || !index) {
      throw new Error(`Unrecognized bit format: '${bitToken}'`);
    }

    const gateId = `g_${this.gateIdCounter++}`;
    const bitId = `${register}${index}`;
    const source = this._registerGateOnBit(bitId, gateId);

    currentGraph.edges.push({ source, target: gateId });
    this._addGateNode(currentGraph, {
      id: gateId,
      type: "one_quit_gate",
      name: gateName,
      gate_info: gateInfo
    });
    return true;
  }

  _handleTwoQubitParamGate(parts, currentGraph, originalLine) {
    const gateName = parts[0].trim();
    const gateInfo = parts[1].trim();
    const bitToken1 = parts[2].trim().replace(/[,;]$/g, "");
    const bitToken2 = parts[3].trim().replace(/;$/, "");
    const [register1, index1] = getBitInfo(bitToken1);
    const [register2, index2] = getBitInfo(bitToken2);
    if (!register1 || !index1 || !register2 || !index2) {
      throw new Error(`Unrecognized bit format in line: '${originalLine}'`);
    }

    const gateId = `g_${this.gateIdCounter++}`;
    const bitId1 = `${register1}${index1}`;
    const bitId2 = `${register2}${index2}`;
    const source1 = this._registerGateOnBit(bitId1, gateId);
    const source2 = this._registerGateOnBit(bitId2, gateId);

    currentGraph.edges.push({ source: source1, target: gateId });
    currentGraph.edges.push({ source: source2, target: gateId });
    this._addGateNode(currentGraph, {
      id: gateId,
      type: "two_qubit_gate",
      name: gateName,
      gate_info: gateInfo
    });
    return true;
  }

  _handleSingleQubitGate(parts, currentGraph) {
    const [gateName, bitTokenRaw] = parts;
    const bitToken = bitTokenRaw.replace(/;$/, "");
    const [register, index] = getBitInfo(bitToken);
    if (!register || !index) {
      throw new Error(`Unrecognized bit format: '${bitToken}'`);
    }

    const gateId = `g_${this.gateIdCounter++}`;
    const bitId = `${register}${index}`;
    const source = this._registerGateOnBit(bitId, gateId);

    currentGraph.edges.push({ source, target: gateId });
    this._addGateNode(currentGraph, {
      id: gateId,
      type: "single_qubit_gate",
      name: gateName
    });
    return true;
  }

  _handleTwoQubitGate(parts, currentGraph, originalLine) {
    const [gateName, bitTokenRaw1, bitTokenRaw2] = parts;
    const bitToken1 = bitTokenRaw1.replace(/[,;]$/g, "");
    const bitToken2 = bitTokenRaw2.replace(/;$/, "");
    const [register1, index1] = getBitInfo(bitToken1);
    const [register2, index2] = getBitInfo(bitToken2);
    if (!register1 || !index1 || !register2 || !index2) {
      throw new Error(`Unrecognized bit format in line: '${originalLine}'`);
    }

    const gateId = `g_${this.gateIdCounter++}`;
    const bitId1 = `${register1}${index1}`;
    const bitId2 = `${register2}${index2}`;
    const source1 = this._registerGateOnBit(bitId1, gateId);
    const source2 = this._registerGateOnBit(bitId2, gateId);

    currentGraph.edges.push({ source: source1, target: gateId });
    currentGraph.edges.push({ source: source2, target: gateId });
    this._addGateNode(currentGraph, {
      id: gateId,
      type: "two_qubit_gate",
      name: gateName
    });
    return true;
  }

  _handleMeasureGate(parts, currentGraph) {
    const [, quantumBitRaw, , classicalBitRaw] = parts;
    const quantumToken = quantumBitRaw.replace(/[,;]$/g, "");
    const classicalToken = classicalBitRaw.replace(/;$/, "");

    const [quantumRegister, quantumIndex] = getBitInfo(quantumToken);
    const [classicalRegister, classicalIndex] = getBitInfo(classicalToken);

    if (!quantumRegister || !quantumIndex || !classicalRegister || !classicalIndex) {
      throw new Error(`Unrecognized bit format in measurement: '${parts.join(" ")}'`);
    }

    const gateId = `g_${this.gateIdCounter++}`;
    const quantumBitId = `${quantumRegister}${quantumIndex}`;
    const classicalBitId = `${classicalRegister}${classicalIndex}`;
    const quantumSource = this._registerGateOnBit(quantumBitId, gateId);
    const classicalSource = this._registerGateOnBit(classicalBitId, gateId);

    currentGraph.edges.push({ source: quantumSource, target: gateId });
    currentGraph.edges.push({ source: classicalSource, target: gateId });
    this._addGateNode(currentGraph, {
      id: gateId,
      type: "measurement",
      name: "measure"
    });
    return true;
  }
}

function convertQasmToGraphJson(qasmText) {
  const parser = new QasmParser(qasmText);
  return parser.parse();
}

function getGraphDimensions() {
  const svgElement = document.getElementById("graph");

  if (!svgElement) {
    return {
      width: window.innerWidth || 928,
      height: window.innerHeight || 600
    };
  }

  const rect = svgElement.getBoundingClientRect();

  return {
    width: Math.max(rect.width, 300),
    height: Math.max(rect.height, 300)
  };
}

async function loadGraphData(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load graph data: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

function createSimulation(nodes, links) {
  return d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(links)
        .id(d => d.id)
        .distance(80)
    )
    .force("charge", d3.forceManyBody().strength(-50))
    .force("center", d3.forceCenter(0, 0))
    .force("x", d3.forceX(0).strength(0.025))
    .force("y", d3.forceY(0).strength(0.025))
    .force("collision", d3.forceCollide().radius(20));
}

function initSvg(width, height) {
  const svg = d3
    .select("#graph")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", [-width / 2, -height / 2, width, height].join(" "))
    .attr("style", "width: 100%; height: 100%; display: block;");

  // Clear any previous render.
  svg.selectAll("*").remove();

  return svg;
}

function renderGraph(data) {
  const { width, height } = getGraphDimensions();

  const nodes = data?.nodes?.map(d => ({ ...d })) ?? [];
  const links = data?.edges?.map(d => ({ ...d })) ?? [];

  if (!nodes.length) {
    throw new Error("Graph data is missing nodes.");
  }

  hideNodeTooltip();

  const simulation = createSimulation(nodes, links);
  const svg = initSvg(width, height);
  const viewport = svg.append("g").attr("class", "graph-viewport");
  viewport.attr("transform", currentTransform);

  const zoomBehavior = d3
    .zoom()
    .scaleExtent([0.25, 6])
    .on("zoom", event => {
      currentTransform = event.transform;
      viewport.attr("transform", currentTransform);
    });

  svg.call(zoomBehavior).on("dblclick.zoom", null);
  svg.call(zoomBehavior.transform, currentTransform);

  const link = viewport
    .append("g")
    .attr("stroke", "#999")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke-width", d => {
      const baseWidth = Math.sqrt(d.value ?? 2) * 4;
      d.__baseWidth = baseWidth;
      d.__baseColor = "#999";
      return d.__baseWidth;
    });

  const node = viewport
    .append("g")
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5)
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", 12)
    .attr("fill", d => {
      d.__baseColor = getNodeColor(d.type);
      return d.__baseColor;
    })
    .call(
      d3
        .drag()
        .on("start", event => dragstarted(event, simulation))
        .on("drag", dragged)
        .on("end", event => dragended(event, simulation))
    );

  node
    .on("mouseenter", (event, target) => {
      setHighlightState(target, true);
      if (target.type === "one_quit_gate") {
        showNodeTooltip(event, target);
      } else {
        hideNodeTooltip();
      }
    })
    .on("mousemove", (event, target) => {
      if (target.type === "one_quit_gate") {
        positionNodeTooltip(event);
      }
    })
    .on("mouseleave", (event, target) => {
      setHighlightState(target, false);
      hideNodeTooltip();
    });

  node.append("title").text(d => {
    const labelParts = [d.name ?? d.id];
    if (d.type) {
      labelParts.push(`type: ${d.type}`);
    }
    return labelParts.join("\n");
  });

  updateLegend(nodes);

  const label = viewport
    .append("g")
    .selectAll("text")
    .data(nodes)
    .join("text")
    .attr("fill", "#222")
    .attr("font-size", 10)
    .attr("text-anchor", "middle")
    .attr("dy", "2.2em")
    .text(d => d.name ?? d.id);

  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

    node.attr("cx", d => d.x).attr("cy", d => d.y);

    label.attr("x", d => d.x).attr("y", d => d.y);
  });

  function setHighlightState(targetNode, isActive) {
    node
      .attr("fill", d => d.__baseColor)
      .attr("stroke", d => (isActive && d === targetNode ? "#000" : "#fff"))
      .attr("stroke-width", d => (isActive && d === targetNode ? 3 : 1.5));

    label.attr("font-weight", "400");

    link
      .attr("stroke", d => {
        if (!isActive) {
          return d.__baseColor;
        }
        return linkTouchesNode(d, targetNode) ? "#000" : d.__baseColor;
      })
      .attr("stroke-width", d => {
        if (!isActive) {
          return d.__baseWidth ?? 1.5;
        }
        return linkTouchesNode(d, targetNode) ? Math.max(d.__baseWidth ?? 1.5, 3) : d.__baseWidth ?? 1.5;
      })
      .attr("stroke-opacity", 0.6);
  }

  function linkTouchesNode(linkDatum, nodeDatum) {
    const sourceId = linkDatum.source?.id ?? linkDatum.source;
    const targetId = linkDatum.target?.id ?? linkDatum.target;
    return sourceId === nodeDatum.id || targetId === nodeDatum.id;
  }
}

function getTimeSteps(data) {
  if (!data || typeof data !== "object") {
    return [];
  }

  return Object.keys(data)
    .map(key => Number(key))
    .filter(step => Number.isFinite(step))
    .sort((a, b) => a - b);
}

function hasOwnProperty(target, key) {
  return Object.prototype.hasOwnProperty.call(target ?? {}, key);
}

function getTimeStepData(data, step) {
  if (!data) {
    return null;
  }

  if (hasOwnProperty(data, step)) {
    return data[step];
  }

  const stringKey = String(step);
  if (hasOwnProperty(data, stringKey)) {
    return data[stringKey];
  }

  return null;
}

function updateTimeStepLabel(step) {
  const label = document.getElementById("time-step-value");
  if (label) {
    label.textContent = String(step);
  }
}

function syncSliderValue(step) {
  const slider = document.getElementById("time-slider");
  if (!slider || availableTimeSteps.length === 0) {
    return;
  }

  const index = availableTimeSteps.indexOf(step);
  if (index >= 0) {
    slider.value = String(index);
    slider.setAttribute("aria-valuetext", String(step));
  }
}

function renderTimeStep(step) {
  const timestepData = getTimeStepData(cachedGraphData, step);
  if (!timestepData) {
    console.warn(`Graph data does not contain time step ${step}.`);
    return;
  }
  renderGraph(timestepData);
}

function initializeGraphFromData(graphData) {
  cachedGraphData = graphData;
  availableTimeSteps = getTimeSteps(cachedGraphData);

  if (!availableTimeSteps.length) {
    throw new Error("Graph data does not contain any time steps.");
  }

  pauseAnimation(false);
  currentTransform = d3.zoomIdentity;
  currentTimeStep = null;

  const initialStep = availableTimeSteps.includes(DEFAULT_TIME_STEP)
    ? DEFAULT_TIME_STEP
    : availableTimeSteps[0];

  setupTimeSlider(initialStep);
  setCurrentTimeStep(initialStep);
  updatePlayButtonAvailability();
  updatePlayButtonUI(false);
}

function setCurrentTimeStep(step) {
  if (!availableTimeSteps.includes(step)) {
    console.warn(`Time step ${step} is not available in the dataset.`);
    return;
  }

  currentTimeStep = step;
  updateTimeStepLabel(step);
  syncSliderValue(step);
  renderTimeStep(step);
}

function handleTimeSliderInput(event) {
  const index = Number(event.target.value);
  if (!Number.isInteger(index) || index < 0 || index >= availableTimeSteps.length) {
    return;
  }

  pauseAnimation();
  const nextStep = availableTimeSteps[index];
  if (typeof nextStep === "undefined") {
    return;
  }

  setCurrentTimeStep(nextStep);
}

function setupTimeSlider(initialStep) {
  const slider = document.getElementById("time-slider");
  if (!slider) {
    return;
  }

  slider.min = "0";
  slider.max = String(Math.max(availableTimeSteps.length - 1, 0));
  slider.step = "1";
  slider.disabled = availableTimeSteps.length <= 1;

  slider.removeEventListener("input", handleTimeSliderInput);
  slider.addEventListener("input", handleTimeSliderInput);

  syncSliderValue(initialStep);
  updatePlayButtonAvailability();
}

function dragstarted(event, simulation) {
  if (!event.active) simulation.alphaTarget(0.3).restart();
  event.subject.fx = event.subject.x;
  event.subject.fy = event.subject.y;
}

function dragged(event) {
  event.subject.fx = event.x;
  event.subject.fy = event.y;
}

function dragended(event, simulation) {
  if (!event.active) simulation.alphaTarget(0);
  event.subject.fx = null;
  event.subject.fy = null;
}

function handleResize() {
  if (resizeFrame !== null) {
    return;
  }

  resizeFrame = window.requestAnimationFrame(() => {
    resizeFrame = null;
    if (currentTimeStep != null) {
      renderTimeStep(currentTimeStep);
    }
  });
}

function isAnimationActive() {
  return animationTimerId !== null;
}

function updatePlayButtonUI(isPlaying) {
  const button = document.getElementById("play-toggle");
  if (!button) {
    return;
  }

  button.innerHTML = isPlaying ? "&#10072;&#10072;" : "&#9654;";
  button.setAttribute("aria-label", isPlaying ? "Pause animation" : "Play animation");
  button.setAttribute("aria-pressed", String(isPlaying));
}

function pauseAnimation(updateUI = true) {
  if (!isAnimationActive()) {
    if (updateUI) {
      updatePlayButtonUI(false);
    }
    return;
  }

  window.clearInterval(animationTimerId);
  animationTimerId = null;

  if (updateUI) {
    updatePlayButtonUI(false);
  }
}

function getNextTimeStep(step) {
  if (!availableTimeSteps.length) {
    return null;
  }

  const currentIndex = availableTimeSteps.indexOf(step);
  const nextIndex = currentIndex >= 0 ? (currentIndex + 1) % availableTimeSteps.length : 0;
  return availableTimeSteps[nextIndex] ?? null;
}

function advanceToNextTimeStep() {
  if (availableTimeSteps.length === 0) {
    pauseAnimation();
    return;
  }

  const nextStep = getNextTimeStep(currentTimeStep ?? availableTimeSteps[0]);
  if (nextStep != null) {
    setCurrentTimeStep(nextStep);
  }
}

function startAnimation() {
  if (isAnimationActive() || availableTimeSteps.length <= 1) {
    updatePlayButtonUI(false);
    return;
  }

  if (currentTimeStep == null) {
    const initialStep = availableTimeSteps[0];
    if (initialStep != null) {
      setCurrentTimeStep(initialStep);
    }
  }

  updatePlayButtonUI(true);

  animationTimerId = window.setInterval(() => {
    advanceToNextTimeStep();
  }, ANIMATION_INTERVAL_MS);
}

function toggleAnimation() {
  if (isAnimationActive()) {
    pauseAnimation();
  } else {
    startAnimation();
  }
}

function updatePlayButtonAvailability() {
  const button = document.getElementById("play-toggle");
  if (!button) {
    return;
  }

  const shouldDisable = availableTimeSteps.length <= 1;
  button.disabled = shouldDisable;

  if (shouldDisable) {
    pauseAnimation(false);
    updatePlayButtonUI(false);
  }
}

function handleFileUpload(event) {
  const [file] = event.target?.files ?? [];
  if (!file) {
    return;
  }

  const reader = new FileReader();

  reader.onload = loadEvent => {
    try {
      const text = typeof loadEvent.target?.result === "string" ? loadEvent.target.result : "";
      const parsedData = JSON.parse(text);
      initializeGraphFromData(parsedData);
    } catch (error) {
      console.error("Failed to parse uploaded graph JSON.", error);
      window.alert("We couldn't read that JSON file. Please ensure it is valid time-dependent graph data.");
    } finally {
      event.target.value = "";
    }
  };

  reader.onerror = error => {
    console.error("Failed to read uploaded file.", error);
    window.alert("We couldn't read that file. Please try again or choose a different file.");
    event.target.value = "";
  };

  reader.readAsText(file);
}

function handleQasmUpload(event) {
  const [file] = event.target?.files ?? [];
  if (!file) {
    return;
  }

  const reader = new FileReader();

  reader.onload = loadEvent => {
    try {
      const qasmText = typeof loadEvent.target?.result === "string" ? loadEvent.target.result : "";
      if (!qasmText.trim()) {
        throw new Error("File was empty.");
      }
      const graphJson = convertQasmToGraphJson(qasmText);
      initializeGraphFromData(graphJson);

      const downloadBlob = new Blob([JSON.stringify(graphJson, null, 2)], {
        type: "application/json"
      });
      const url = URL.createObjectURL(downloadBlob);
      const anchor = document.createElement("a");
      anchor.href = url;
      const safeName = file.name.replace(/\.[^./\\]+$/, "");
      anchor.download = `${safeName || "converted"}-graph.json`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Failed to convert QASM file.", error);
      window.alert("We couldn't convert that QASM file. Please ensure it is valid OpenQASM.");
    } finally {
      event.target.value = "";
    }
  };

  reader.onerror = error => {
    console.error("Failed to read uploaded file.", error);
    window.alert("We couldn't read that file. Please try again or choose a different file.");
    event.target.value = "";
  };

  reader.readAsText(file);
}

async function boot() {
  try {
    const initialGraph = await loadGraphData(GRAPH_PATH);
    initializeGraphFromData(initialGraph);
  } catch (error) {
    console.error(error);
  }

  window.addEventListener("resize", handleResize);
  const playToggle = document.getElementById("play-toggle");
  if (playToggle) {
    playToggle.addEventListener("click", toggleAnimation);
  }

  const fileUpload = document.getElementById("file-upload");
  if (fileUpload) {
    fileUpload.addEventListener("change", handleFileUpload);
  }

  const qasmUpload = document.getElementById("qasm-upload");
  if (qasmUpload) {
    qasmUpload.addEventListener("change", handleQasmUpload);
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      pauseAnimation();
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}