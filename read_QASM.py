from qiskit import QuantumCircuit

def read_QASM(filename):
    with open(filename, 'r') as file:
        qasm_code = file.read()
    return qasm_code

def main():
    filename = 'test.qasm'
    qasm_code = read_QASM(filename)
    print(qasm_code)

if __name__ == '__main__':
    main()
    