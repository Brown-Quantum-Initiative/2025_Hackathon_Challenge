import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit.qasm2 import dump
import os
# print(qiskit.__version__)

# converts qasm file to a a QuantumCircuit object
def read_qasm(qasm_filepath):
  original_circuit = QuantumCircuit.from_qasm_file(qasm_filepath)
  return original_circuit

current_directory = os.getcwd()
# print(current_directory)

original_circuit = read_qasm(current_directory + '/big_circuits/1.qasm') # NOTE add desired file here

two_qubit_gate_count = 0 # counter for initially how many 2-qubit gates there are
higher_qubit_gate_count = 0
for instruction in original_circuit.data:
    # Check if the number of qubits the instruction acts on is 2
    if len(instruction.qubits) == 2:
        two_qubit_gate_count += 1
    elif len(instruction.qubits) > 2:
        higher_qubit_gate_count += 1

print('##########################################')
print("In input circuit: ")
print(f"Number of 2-qubit gates: {two_qubit_gate_count}")
print(f"Number of (>2)-qubit gates: {higher_qubit_gate_count}")
print('##########################################')

# print(original_circuit)

# transpilation to fully-connected graph with no restrictions on basis states
qc_all = transpile(original_circuit, coupling_map=None, basis_gates=['rzz', 'rx', 'ry', 'rz'], optimization_level=3)
# print(qc_all)

backend_fez = FakeFez() # backend associated with the topology of IBM Fez

# transpilation to fez topology and basis states
qc_fez = transpile(
    original_circuit,
    backend=backend_fez,
    optimization_level=3,
    layout_method='sabre',
    routing_method='sabre'
)

# print(qc_fez)

# writes transpiled QuantumCircuit object out as a .qasm files for each case
with open("all_qc.qasm", "w") as f:
    dump(qc_all, f) # fully connected case
with open("fez_qc.qasm", "w") as f:
    dump(qc_fez, f) # IBM Fez case


qc_all_2qb_count = 0 # all 2 qubit gates in the fully connected transpiled circuit
qc_all_higher_count = 0 # count of all (>2) qubit gates
for instruction in qc_all.data:
    # Check if the number of qubits the instruction acts on is 2
    if len(instruction.qubits) == 2:
        qc_all_2qb_count += 1
    elif len(instruction.qubits) > 2:
        qc_all_higher_count += 1

print('##########################################')
print("In fully connected circuit: ")
print(f"Number of 2-qubit gates: {qc_all_2qb_count}")
print(f"Number of (>2)-qubit gates: {qc_all_higher_count}")
print('##########################################')

qc_fez_count = 0 # counts all 2-qubit gates in circuit transpiled for Fez
qc_fez_higher = 0 # counts all (>2) qubit gates in transpiled circuit
for instruction in qc_fez.data:
    # Check if the number of qubits the instruction acts on is 2
    if len(instruction.qubits) == 2:
        qc_fez_count += 1
    elif len(instruction.qubits) > 2:
        qc_fez_higher += 1

print('##########################################')
print("In Fez appropriate circuit: ")
print(f"Number of 2-qubit gates: {qc_fez_count}")
print(f"Number of (>2)-qubit gates: {qc_fez_higher}")
print('##########################################')


# Print the number of qubits to diagnose the issue
print(f"Number of qubits in original_circuit: {original_circuit.num_qubits}")
print(f"Number of qubits in qc_all: {qc_all.num_qubits}")
print(f"Number of qubits in qc_fez: {qc_fez.num_qubits}")