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
        self.gate_id_counter = 0
        self.gates = []
        self.bits = {}
        self.links = []
        self.timestamps = []

    def get_bits(self):
        with open(self.qasm_file, 'r') as file:
            self.qasm_code = file.read()
        
        for raw_line in self.qasm_code.split('\n'):
            line = raw_line.split('//')[0].strip()
            if not line:
                continue

            if line.startswith('qreg'):  
                match = re.search(r'qreg\s+\w+\[(\d+)\]', line)
                if match:
                    num_qubits = int(match.group(1))
                for i in range(num_qubits):
                    self.bits[f'q{i}'] = {"id": f'q{i}', 
                                          "type": "qubit", 
                                          "name": f'q{i}', 
                                          "last_gate_connected": None}
            if line.startswith('creg'):
                match = re.search(r'creg\s+\w+\[(\d+)\]', line)
                if match:
                    num_classical_bits = int(match.group(1))
                    for i in range(num_classical_bits):
                        self.bits[f'c{i}'] = {"id": f'c{i}', 
                                              "type": "classical_bit", 
                                              "name": f'c{i}', 
                                              "last_gate_connected": None}
        return self.bits

    def get_gates(self):
        
        for raw_line in self.qasm_code.split('\n'):
            line = raw_line.split('//')[0].strip()
            if not line:
                continue

            if line.startswith('//'): continue
            if line.startswith('OPENQASM'): continue
            if line.startswith('include'): continue
            if line.startswith('qreg'): continue
            if line.startswith('creg'): continue
            # Remove comments and strip line
            
            if "(" in line and ")" in line:
                parts = re.split(r'[()]', line)
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
                    #print(gate_type, gate_name, gate_info, letter1, number1, letter2, number2)

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
                    
                else:
                    continue
            else: 
                parts = re.split(r'[,\s]+', line)
                if len(parts) == 2: 
                    gate_type = "single_qubit_gate"
                    gate_name = parts[0]
                    bit = parts[1].strip().rstrip(';')
                    letter, number = get_bit_info(bit)
                    if letter is None or number is None:
                        raise ValueError(f"Unrecognized bit format: '{bit}'")
                    #print(gate_type, gate_name, letter, number)
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
                    

                elif len(parts) == 3:
                    gate_type = "two_qubit_gate"
                    gate_name = parts[0]
                    bit1 = parts[1].strip().rstrip(';')
                    bit2 = parts[2].strip().rstrip(';')
                    letter1, number1 = get_bit_info(bit1)
                    letter2, number2 = get_bit_info(bit2)
                    if None in (letter1, number1, letter2, number2):
                        raise ValueError(f"Unrecognized bit format in line: '{line}'")
                    #print(gate_type, gate_name, letter1, number1, letter2, number2)
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
                    
                else:
                    continue 
                    
                
                

            


if __name__ == "__main__":
    parser = QASM_Parser("parser/5.qasm")
    parser.get_bits()
    parser.get_gates()
    print(parser.bits)
    print(parser.links)