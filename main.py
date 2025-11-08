from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit.qasm2 import QASM2ParseError

import numpy as np
import matplotlib
import re

from pathlib import Path
from google.colab import drive

class OptimizedTranspiler:
    def __init__(self, backend, basis_gates, seed_range):
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
        """
        ops = circuit.count_ops()
        twoq_gates = ["cx", "cz", "iswap", "ecr", "rzz"]
        return sum(ops.get(g, 0) for g in twoq_gates)
