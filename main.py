from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime.fake_provider import FakeFez
import qiskit.qasm3
from qiskit.qasm3 import loads as qasm3_loads
import numpy as np
import matplotlib
import re

from pathlib import Path
from math import inf

class OptimizedTranspiler:
    def __init__(self, backend=None, basis_gates=None, seed_range=None):
        """
        An optimized transpiler class that allows you to configure seed and gate basis settings.
        Param backend: set the transpiler backend, default = FakeFez()
        Param basis_gates: set the basis gates you want to try, default = Fez default gates
        Param seed_range: set the range of seeds to try over, default = (0, 100)
        """
        self.backend = backend
        if self.backend is None:
            self.backend = FakeFez()
        
        self.basis_gates = basis_gates
        if self.basis_gates is None:
            self.basis_gates = {"default": ['cz', 'id', 'rz', 'sx', 'x']}

        self.seed_range = seed_range
        if self.seed_range is None:
            self.seed_range = (0, 100)

        self.prev_circ = None

    def try_gates(self, circuit, gates=None, seed=0):
        """
        Transpile the given circuit on all possible given basis gates.
        Param circuit: the circuit to transpile
        Param gates: the basis gates to try transpiling with. Default: use self.basis_gates
        Param seed: transpile with a custom seed. Default: 0.
        Returns: Dictionary from gate name to two qubit gate count
        """
        self.prev_circ = circuit
        if not gates:
            gates = self.basis_gates

        out = {} 
        for name in gates:
            basis = gates[name]
            transpiled = transpile(circuit, self.backend, basis_gates=basis, optimization_level=3, seed_transpiler=seed)
            two_qubit_gates = self.two_qubit_gate_count()
            out[name] = two_qubit_gates

        return out

    def try_seeds(self, circuit, gates=None, seed=None):
        """
        Transpile the given circuit on all possible given seeds.

        Param circuit: the circuit to transpile
        Param gates: dict name -> list of basis gates. Default: single 'default' basis.
        Param seed: range or (start, stop) tuple of seeds to try.
                    Default: use self.seed_range.
        Returns: the best seed (int), chosen to minimize (two-qubit gates, depth).
        """
        if gates is None:
            gates = {"default": ['cz', 'id', 'rz', 'sx', 'x']}

        if seed is None:
            seed_range = self.seed_range   # e.g. (0, 32) or range(...)
        else:
            seed_range = seed

        if isinstance(seed_range, tuple):
            start, stop = seed_range
            seed_iter = range(start, stop)
        else:
            seed_iter = seed_range  # assume it's already iterable

        best_seed = None
        best_2q = inf
        best_depth = inf

        # sweep seeds; for each seed, try all basis sets in gates
        for s in seed_iter:
            for _, basis in gates.items():
                transpiled = transpile(
                    circuit,
                    self.backend,
                    basis_gates=basis,
                    optimization_level=3,
                    seed_transpiler=s,
                )
                two_q = self.two_qubit_gate_count(transpiled)
                depth = transpiled.depth()

                # same strategy as in optimize_for_fez: minimize (2Q, depth)
                if (two_q < best_2q) or (two_q == best_2q and depth < best_depth):
                    best_2q = two_q
                    best_depth = depth
                    best_seed = s

        return best_seed

    def two_qubit_gate_count(self, circuit):
        """
        Get two qubit gate count.
        """
        ops = circuit.count_ops()
        twoq_gates = ["cx", "cz", "iswap", "ecr", "rzz"]
        return sum(ops.get(g, 0) for g in twoq_gates)

def robust_load_qasm(path: Path) -> QuantumCircuit:
    """
    Load a QASM file and automatically patch common issues for QASM 2.0,
    or parse directly for QASM 3.0.
    """
    path = Path(path)
    text = path.read_text(encoding='utf-8') # Specify encoding to avoid potential errors

    if "OPENQASM 3.0;" in text:
        try:
            return qasm3_loads(text)
        except Exception as e:
            print(f"Error loading QASM 3.0 file {path}: {e}")
            raise
    else:
        # Assume QASM 2.0 and apply patching
        # Ensure a valid header
        if "OPENQASM" not in text:
            header = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            text = header + text
        elif 'include "qelib1.inc";' not in text:
            first_newline = text.find("\n")
            if first_newline == -1:
                first_newline = len(text)
            text = text[:first_newline+1] + 'include "qelib1.inc";\n' + text[first_newline+1:]

        # Insert qreg if missing (only relevant for QASM 2.0)
        if "qreg" not in text:
            indices = [int(m.group(1)) for m in re.finditer(r"q\[(\d+)\]", text)]
            n_qubits = max(indices) + 1 if indices else 1
            # Find insertion point after includes
            insert_pos = -1
            if 'include "qelib1.inc";' in text:
                insert_pos = text.find('include "qelib1.inc";')
            elif 'OPENQASM 2.0;' in text:
                insert_pos = text.find('OPENQASM 2.0;')

            if insert_pos != -1:
                insert_pos = text.find("\n", insert_pos) + 1
            else: # Fallback if headers are completely missing and patched at start
                insert_pos = text.find("\n") + 1 # After the first line

            if insert_pos == 0: # If there's only one line or no newlines at all
                insert_pos = len(text)

            text = text[:insert_pos] + f"qreg q[{n_qubits}];\n" + text[insert_pos:]
        try:
            return QuantumCircuit.from_qasm_str(text)
        except Exception as e:
            print(f"Error loading QASM 2.0 file {path}: {e}")
            raise

ot = OptimizedTranspiler()
qc1 = robust_load_qasm("big_circuits/1.qasm")
qc2 = robust_load_qasm("big_circuits/2.qasm")
qc3 = robust_load_qasm("big_circuits/3.qasm")

qc1_seed = ot.try_seeds(circuit=qc1, seed=(0,1000))
qc2_seed = ot.try_seeds(circuit=qc2, seed=(0,1000))
qc3_seed = ot.try_seeds(circuit=qc3, seed=(0,1000))

final1 = transpile(qc1, backend=FakeFez(), optimization_level=3, seed_transpiler=qc1_seed)
final2 = transpile(qc2, backend=FakeFez(), optimization_level=3, seed_transpiler=qc2_seed)
final3 = transpile(qc3, backend=FakeFez(), optimization_level=3, seed_transpiler=qc3_seed)
