import copy
import re

def get_bit_info(bit):
    match = re.match(r'([a-zA-Z]+)\[(\d+)\]', bit)
    if match:
        letter = match.group(1)
        number = match.group(2)
        return letter, number
    else:
        return None, None

class QASM_Parser:
    def __init__(self, qasm_file):
        self.qasm_file = qasm_file
        self.version = None
        self.gate_id_counter = 0
        self.gates = []
        self.bits = {}
        self.links = []
        self.timestamps = {}

    def get_bits(self):
        with open(self.qasm_file, 'r') as file:
            self.qasm_code = file.read()
        
        for raw_line in self.qasm_code.split('\n'):
            line = raw_line.split('//')[0].strip()
            if not line:
                continue
            if line.startswith('OPENQASM'):
                parts = line.split()
                if len(parts) > 1:
                    self.version = parts[1].rstrip(';')
                break

        for raw_line in self.qasm_code.split('\n'):
            graph_updated = False
            line = raw_line.split('//')[0].strip()
            if not line:
                continue

            if self.version is None:
                continue
            
            if self.version == "2.0":
                if line.startswith('qreg'):
                    match = re.search(r'qreg\s+([a-zA-Z_]\w*)\[(\d+)\]', line)
                    if match:
                        register_name = match.group(1)
                        num_qubits = int(match.group(2))
                        for i in range(num_qubits):
                            bit_id = f'{register_name}{i}'
                            self.bits[bit_id] = {
                                "id": bit_id,
                                "type": "qubit",
                                "name": bit_id,
                                "last_gate_connected": None
                            }
                if line.startswith('creg'):
                    match = re.search(r'creg\s+([a-zA-Z_]\w*)\[(\d+)\]', line)
                    if match:
                        register_name = match.group(1)
                        num_classical_bits = int(match.group(2))
                        for i in range(num_classical_bits):
                            bit_id = f'{register_name}{i}'
                            self.bits[bit_id] = {
                                "id": bit_id,
                                "type": "classical_bit",
                                "name": bit_id,
                                "last_gate_connected": None
                            }

            elif self.version == "3.0":
                if line.startswith('qubit'):
                    match_name_first = re.search(r'qubit\s+([a-zA-Z_]\w*)\[(\d+)\]', line)
                    match_size_first = re.search(r'qubit\[(\d+)\]\s+([a-zA-Z_]\w*)', line)

                    if match_name_first:
                        register_name = match_name_first.group(1)
                        size = int(match_name_first.group(2))
                    elif match_size_first:
                        size = int(match_size_first.group(1))
                        register_name = match_size_first.group(2)
                    else:
                        continue

                    for i in range(size):
                        bit_id = f'{register_name}{i}'
                        self.bits[bit_id] = {
                            "id": bit_id,
                            "type": "qubit",
                            "name": bit_id,
                            "last_gate_connected": None
                        }
        return self.bits

    def get_gates(self):
        
        line_counter = 0
        current_graph = {"nodes": list(self.bits.values()), "edges": []}
        self.timestamps[0] = copy.deepcopy(current_graph)

        for raw_line in self.qasm_code.split('\n'):
            line = raw_line.split('//')[0].strip()
            if not line:
                continue

            if line.startswith('//'): continue
            if line.startswith('OPENQASM'): continue
            if line.startswith('include'): continue

            if self.version == "2.0":
                if line.startswith('qreg'): continue
                if line.startswith('creg'): continue
                if 'qubit' in line: continue

            elif self.version == "3.0":
                if 'qubit' in line: continue

            line_counter += 1

            # ------------------------------------------------
            # gates with inputs: one-qubit gate, two-qubit gate
            # ------------------------------------------------
            if "(" in line and ")" in line:
                parts = re.split(r'[()]', line)

                # ------------------------------------------------
                # one-qubit gate
                # ------------------------------------------------
                if len(parts) == 3:
                    gate_type = "one_quit_gate"
                    gate_name = parts[0]
                    gate_info = parts[1]
                    bit = parts[2].strip().rstrip(';')
                    
                    letter, number = get_bit_info(bit)
                    
                    if letter is None or number is None:
                        raise ValueError(f"Unrecognized bit format: '{bit}'")

                    gate_id = f"g_{self.gate_id_counter}"
                    self.gates.append({"id": gate_id, 
                                        "type": gate_type, 
                                        "name": gate_name, 
                                        "gate_info": gate_info})

                    self.gate_id_counter += 1
                    
                    last_gate_connected = self.bits[f"{letter}{number}"]["last_gate_connected"]
                    if last_gate_connected is None:
                        source = f"{letter}{number}"
                    else: 
                        source = last_gate_connected
                    
                    self.links.append({"source": source, "target": gate_id})
                    self.bits[f"{letter}{number}"]["last_gate_connected"] = gate_id   
            
                    # ------------------------------------------------
                    current_graph["nodes"].append({"id": gate_id, 
                                                        "type": gate_type, 
                                                        "name": gate_name, 
                                                        "gate_info": gate_info})
                    current_graph["edges"].append({"source": source, "target": gate_id})
                    graph_updated = True
                    # ------------------------------------------------

                # ------------------------------------------------
                # two-qubit gate
                # ------------------------------------------------
                elif len(parts) == 4:
                    gate_type = "two_qubit_gate"
                    gate_name = parts[0]
                    gate_info = parts[1]
                    bit1 = parts[2].strip().rstrip(';')
                    bit2 = parts[3].strip().rstrip(';')
                    letter1, number1 = get_bit_info(bit1)
                    letter2, number2 = get_bit_info(bit2)
                    if None in (letter1, number1, letter2, number2):
                        raise ValueError(f"Unrecognized bit format in line: '{line}'")

                    gate_id = f"g_{self.gate_id_counter}"
                    self.gates.append({"id": gate_id, "type": gate_type, "name": gate_name})
                    self.gate_id_counter += 1

                    last_gate_connected1 = self.bits[f"{letter1}{number1}"]["last_gate_connected"]
                    last_gate_connected2 = self.bits[f"{letter2}{number2}"]["last_gate_connected"]
                    if last_gate_connected1 is None:
                        source1 = f"{letter1}{number1}"
                    else: 
                        source1 = last_gate_connected1
                    if last_gate_connected2 is None:
                        source2 = f"{letter2}{number2}"
                    else: 
                        source2 = last_gate_connected2
                    
                    self.links.append({"source": source1, "target": gate_id})
                    self.links.append({"source": source2, "target": gate_id})
                    self.bits[f"{letter1}{number1}"]["last_gate_connected"] = gate_id
                    self.bits[f"{letter2}{number2}"]["last_gate_connected"] = gate_id
                    
                    # ------------------------------------------------
                    current_graph["nodes"].append({"id": gate_id, 
                                                        "type": gate_type, 
                                                        "name": gate_name, 
                                                        "gate_info": gate_info})
                    current_graph["edges"].append({"source": source1, "target": gate_id})
                    current_graph["edges"].append({"source": source2, "target": gate_id})
                    graph_updated = True
                    # ------------------------------------------------
    
                else:
                    continue
            # ------------------------------------------------
            # gates without inputs: single-qubit gate, two-qubit gate
            # ------------------------------------------------
            else: 
                parts = re.split(r'[,\s]+', line)

                # ------------------------------------------------
                # single-qubit gate
                # ------------------------------------------------
                if len(parts) == 2: 
                    gate_type = "single_qubit_gate"
                    gate_name = parts[0]
                    bit = parts[1].strip().rstrip(';')
                    letter, number = get_bit_info(bit)
                    
                    if letter is None or number is None:
                        print(line)
                        raise ValueError(f"Unrecognized bit format: '{bit}'")

                    gate_id = f"g_{self.gate_id_counter}"
                    self.gates.append({"id": gate_id, 
                                       "type": gate_type, 
                                       "name": gate_name})
                    self.gate_id_counter += 1

                    last_gate_connected = self.bits[f"{letter}{number}"]["last_gate_connected"]
                    if last_gate_connected is None:
                        source = f"{letter}{number}"
                    else: 
                        source = last_gate_connected
                    
                    self.links.append({"source": source, 
                                        "target": gate_id})
                    self.bits[f"{letter}{number}"]["last_gate_connected"] = gate_id

                    # ------------------------------------------------
                    current_graph["nodes"].append({"id": gate_id, 
                                                        "type": gate_type, 
                                                        "name": gate_name, 
                                                        })
                    current_graph["edges"].append({"source": source, "target": gate_id})
                    graph_updated = True
                    # ------------------------------------------------

                # ------------------------------------------------
                # two-qubit gate
                # ------------------------------------------------
                elif len(parts) == 3:
                    gate_type = "two_qubit_gate"
                    gate_name = parts[0]
                    bit1 = parts[1].strip().rstrip(';')
                    bit2 = parts[2].strip().rstrip(';')
                    letter1, number1 = get_bit_info(bit1)
                    letter2, number2 = get_bit_info(bit2)
                    if None in (letter1, number1, letter2, number2):
                        raise ValueError(f"Unrecognized bit format in line: '{line}'")
                    gate_id = f"g_{self.gate_id_counter}"
                    self.gates.append({"id": gate_id, "type": gate_type, "name": gate_name})
                    self.gate_id_counter += 1

                    last_gate_connected1 = self.bits[f"{letter1}{number1}"]["last_gate_connected"]
                    last_gate_connected2 = self.bits[f"{letter2}{number2}"]["last_gate_connected"]
                    if last_gate_connected1 is None:
                        source1 = f"{letter1}{number1}"
                    else: 
                        source1 = last_gate_connected1
                    if last_gate_connected2 is None:
                        source2 = f"{letter2}{number2}"
                    else: 
                        source2 = last_gate_connected2
                    self.links.append({"source": source1, "target": gate_id})
                    self.links.append({"source": source2, "target": gate_id})
                    self.bits[f"{letter1}{number1}"]["last_gate_connected"] = gate_id
                    self.bits[f"{letter2}{number2}"]["last_gate_connected"] = gate_id

                    # ------------------------------------------------
                    current_graph["nodes"].append({"id": gate_id, 
                                                        "type": gate_type, 
                                                        "name": gate_name, 
                                                        })
                    current_graph["edges"].append({"source": source1, "target": gate_id})
                    current_graph["edges"].append({"source": source2, "target": gate_id})
                    graph_updated = True
                    # ------------------------------------------------
                    
                else:
                    continue 

            if graph_updated:
                self.timestamps[line_counter] = copy.deepcopy(current_graph)
                
    def save_json(self, filename):
        import json
        with open(filename, "w") as json_file:
            json.dump(self.timestamps, json_file, indent=4)
                

if __name__ == "__main__":
    parser = QASM_Parser("parser/3.qasm")
    parser.get_bits()
    parser.get_gates()
    parser.save_json("parser/graph3.json")