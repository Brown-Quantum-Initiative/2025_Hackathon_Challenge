import fs from 'fs';

// ------------------------------------------------
// Helper: extract bit info like "q[0]" → ("q", "0")
// ------------------------------------------------
function getBitInfo(bit) {
  const match = bit.match(/([a-zA-Z]+)\[(\d+)\]/);
  if (match) {
    const letter = match[1];
    const number = match[2];
    return [letter, number];
  } else {
    return [null, null];
  }
}

// ------------------------------------------------
// QASM Parser Class
// ------------------------------------------------
class QASM_Parser {
  constructor(qasmFile) {
    this.qasmFile = qasmFile;
    this.gateIdCounter = 0;
    this.gates = [];
    this.bits = {};
    this.links = [];
    this.timestamps = {};
    this.qasmCode = '';
  }

  // ------------------------------------------------
  // Parse bit registers
  // ------------------------------------------------
  getBits() {
    this.qasmCode = fs.readFileSync(this.qasmFile, 'utf-8');

    const lines = this.qasmCode.split('\n');
    for (const rawLine of lines) {
      let graphUpdated = false;
      const line = rawLine.split('//')[0].trim();
      if (!line) continue;

      if (line.startsWith('qreg')) {
        const match = line.match(/qreg\s+\w+\[(\d+)\]/);
        if (match) {
          const numQubits = parseInt(match[1]);
          for (let i = 0; i < numQubits; i++) {
            this.bits[`q${i}`] = {
              id: `q${i}`,
              type: 'qubit',
              name: `q${i}`,
              last_gate_connected: null
            };
          }
        }
      }

      if (line.startsWith('creg')) {
        const match = line.match(/creg\s+\w+\[(\d+)\]/);
        if (match) {
          const numClassicalBits = parseInt(match[1]);
          for (let i = 0; i < numClassicalBits; i++) {
            this.bits[`c${i}`] = {
              id: `c${i}`,
              type: 'classical_bit',
              name: `c${i}`,
              last_gate_connected: null
            };
          }
        }
      }
    }
    return this.bits;
  }

  // ------------------------------------------------
  // Parse gates and build connectivity graph
  // ------------------------------------------------
  getGates() {
    let lineCounter = 0;
    const currentGraph = { nodes: Object.values(this.bits), edges: [] };
    this.timestamps[0] = structuredClone(currentGraph);

    const lines = this.qasmCode.split('\n');

    for (const rawLine of lines) {
      const line = rawLine.split('//')[0].trim();
      if (!line) continue;

      if (
        line.startsWith('//') ||
        line.startsWith('OPENQASM') ||
        line.startsWith('include') ||
        line.startsWith('qreg') ||
        line.startsWith('creg')
      ) continue;

      lineCounter++;
      let graphUpdated = false;

      // ------------------------------------------------
      // Gates with parentheses (parameterized gates)
      // ------------------------------------------------
      if (line.includes('(') && line.includes(')')) {
        const parts = line.split(/[()]/);

        // One-qubit gate
        if (parts.length === 3) {
          const gateType = 'one_quit_gate';
          const gateName = parts[0];
          const gateInfo = parts[1];
          const bit = parts[2].trim().replace(/;$/, '');
          const [letter, number] = getBitInfo(bit);
          if (!letter || !number)
            throw new Error(`Unrecognized bit format: '${bit}'`);

          const gateId = `g_${this.gateIdCounter++}`;
          this.gates.push({ id: gateId, type: gateType, name: gateName, gate_info: gateInfo });

          const bitObj = this.bits[`${letter}${number}`];
          const source = bitObj.last_gate_connected || `${letter}${number}`;
          this.links.push({ source, target: gateId });
          bitObj.last_gate_connected = gateId;

          currentGraph.nodes.push({ id: gateId, type: gateType, name: gateName, gate_info: gateInfo });
          currentGraph.edges.push({ source, target: gateId });
          graphUpdated = true;
        }

        // Two-qubit gate
        else if (parts.length === 4) {
          const gateType = 'two_qubit_gate';
          const gateName = parts[0];
          const gateInfo = parts[1];
          const bit1 = parts[2].trim().replace(/;$/, '');
          const bit2 = parts[3].trim().replace(/;$/, '');
          const [letter1, number1] = getBitInfo(bit1);
          const [letter2, number2] = getBitInfo(bit2);
          if (!letter1 || !number1 || !letter2 || !number2)
            throw new Error(`Unrecognized bit format in line: '${line}'`);

          const gateId = `g_${this.gateIdCounter++}`;
          this.gates.push({ id: gateId, type: gateType, name: gateName });

          const bit1Obj = this.bits[`${letter1}${number1}`];
          const bit2Obj = this.bits[`${letter2}${number2}`];
          const source1 = bit1Obj.last_gate_connected || `${letter1}${number1}`;
          const source2 = bit2Obj.last_gate_connected || `${letter2}${number2}`;

          this.links.push({ source: source1, target: gateId });
          this.links.push({ source: source2, target: gateId });
          bit1Obj.last_gate_connected = gateId;
          bit2Obj.last_gate_connected = gateId;

          currentGraph.nodes.push({ id: gateId, type: gateType, name: gateName, gate_info: gateInfo });
          currentGraph.edges.push({ source: source1, target: gateId });
          currentGraph.edges.push({ source: source2, target: gateId });
          graphUpdated = true;
        }
      }

      // ------------------------------------------------
      // Gates without parentheses (normal gates)
      // ------------------------------------------------
      else {
        const parts = line.split(/[,\s]+/).filter(Boolean);

        // Single-qubit gate
        if (parts.length === 2) {
          const gateType = 'single_qubit_gate';
          const gateName = parts[0];
          const bit = parts[1].trim().replace(/;$/, '');
          const [letter, number] = getBitInfo(bit);
          if (!letter || !number)
            throw new Error(`Unrecognized bit format: '${bit}'`);

          const gateId = `g_${this.gateIdCounter++}`;
          this.gates.push({ id: gateId, type: gateType, name: gateName });

          const bitObj = this.bits[`${letter}${number}`];
          const source = bitObj.last_gate_connected || `${letter}${number}`;
          this.links.push({ source, target: gateId });
          bitObj.last_gate_connected = gateId;

          currentGraph.nodes.push({ id: gateId, type: gateType, name: gateName });
          currentGraph.edges.push({ source, target: gateId });
          graphUpdated = true;
        }

        // Two-qubit gate
        else if (parts.length === 3) {
          const gateType = 'two_qubit_gate';
          const gateName = parts[0];
          const bit1 = parts[1].trim().replace(/;$/, '');
          const bit2 = parts[2].trim().replace(/;$/, '');
          const [letter1, number1] = getBitInfo(bit1);
          const [letter2, number2] = getBitInfo(bit2);
          if (!letter1 || !number1 || !letter2 || !number2)
            throw new Error(`Unrecognized bit format in line: '${line}'`);

          const gateId = `g_${this.gateIdCounter++}`;
          this.gates.push({ id: gateId, type: gateType, name: gateName });

          const bit1Obj = this.bits[`${letter1}${number1}`];
          const bit2Obj = this.bits[`${letter2}${number2}`];
          const source1 = bit1Obj.last_gate_connected || `${letter1}${number1}`;
          const source2 = bit2Obj.last_gate_connected || `${letter2}${number2}`;

          this.links.push({ source: source1, target: gateId });
          this.links.push({ source: source2, target: gateId });
          bit1Obj.last_gate_connected = gateId;
          bit2Obj.last_gate_connected = gateId;

          currentGraph.nodes.push({ id: gateId, type: gateType, name: gateName });
          currentGraph.edges.push({ source: source1, target: gateId });
          currentGraph.edges.push({ source: source2, target: gateId });
          graphUpdated = true;
        }
      }

      if (graphUpdated) {
        this.timestamps[lineCounter] = structuredClone(currentGraph);
      }
    }
  }

  // ------------------------------------------------
  // Save as JSON
  // ------------------------------------------------
  saveJSON(filename) {
    fs.writeFileSync(filename, JSON.stringify(this.timestamps, null, 4), 'utf-8');
  }
}

export { QASM_Parser };