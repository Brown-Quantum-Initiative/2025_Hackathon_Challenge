#!/usr/bin/env python3

import pygame
import numpy as np
import math
import io
from pathlib import Path
from typing import Dict, List
from qiskit import QuantumCircuit
try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except Exception:
    AER_AVAILABLE = False
from qiskit.quantum_info import Statevector, partial_trace
try:
    from qiskit import qasm3
    QASM3_AVAILABLE = True
except Exception:
    QASM3_AVAILABLE = False
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image


pygame.init()


PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def load_circuit(qasm_source: str) -> QuantumCircuit:
    path = Path(qasm_source)
    if path.exists():
        # Read the file
        with open(path, 'r') as f:
            qasm_str = f.read()
        
        # Check if it's QASM 3.0
        if 'OPENQASM 3' in qasm_str or 'qubit[' in qasm_str:
            if not QASM3_AVAILABLE:
                raise ImportError("QASM 3.0 file detected but qiskit.qasm3 not available. Install with: pip install qiskit[qasm3-import]")
            print(f"📋 Detected QASM 3.0 format")
            return qasm3.loads(qasm_str)
   
        if 'qreg' not in qasm_str:
            import re
            qubit_indices = [int(m.group(1)) for m in re.finditer(r'q\[(\d+)\]', qasm_str)]
            if qubit_indices:
                num_qubits = max(qubit_indices) + 1
                lines = qasm_str.split('\n')
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if 'include' in line:
                        new_lines.append(f'qreg q[{num_qubits}];')
                qasm_str = '\n'.join(new_lines)
        
        return QuantumCircuit.from_qasm_str(qasm_str)
    return QuantumCircuit.from_qasm_str(qasm_source)


def bloch_vector(state: Statevector, qubit: int) -> np.ndarray:
    from qiskit.quantum_info import partial_trace
    num_qubits = int(np.log2(len(state.data)))
    
    qubits_to_trace = [q for q in range(num_qubits) if q != qubit]
    
    if qubits_to_trace:
        reduced = partial_trace(state, qubits_to_trace)
        rho = reduced.data
    else:
        rho = np.outer(state.data, np.conj(state.data))

    rx = float(np.real(np.trace(rho @ PAULI_X)))
    ry = float(np.real(np.trace(rho @ PAULI_Y)))
    rz = float(np.real(np.trace(rho @ PAULI_Z)))
    return np.array([rx, ry, rz])


def calculate_entanglement(state: Statevector, qubit: int) -> float:
    from qiskit.quantum_info import partial_trace
    import scipy.linalg
    
    num_qubits = int(np.log2(len(state.data)))
    
    if num_qubits == 1:
        return 0.0
    
    qubits_to_trace = [q for q in range(num_qubits) if q != qubit]
    reduced = partial_trace(state, qubits_to_trace)
    rho = reduced.data

    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  
    
    if len(eigenvalues) == 0:
        return 0.0
    
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    

    return min(entropy, 1.0)


def circuit_bloch_trajectories(circuit: QuantumCircuit, selected_qubits: List[int] = None) -> tuple:
    if not selected_qubits:
        selected_qubits = list(range(min(3, circuit.num_qubits)))
    
    if circuit.num_qubits > 20:
        print(f"🚀 Circuit has {circuit.num_qubits} qubits - routing to ultra-sparse approximation")
        return circuit_bloch_trajectories_ultra_sparse(circuit, selected_qubits)
    elif circuit.num_qubits > 8:
        print(f"⚡ Circuit has {circuit.num_qubits} qubits - attempting MPS simulation...")
        try:
            return circuit_bloch_trajectories_mps(circuit, selected_qubits)
        except Exception as e:
            print(f"⚠️ MPS failed ({e}), falling back to ultra-sparse approximation")
            return circuit_bloch_trajectories_ultra_sparse(circuit, selected_qubits)

    print(f"✅ Using exact simulation for {circuit.num_qubits} qubits")
    state = Statevector.from_label("0" * circuit.num_qubits)
    histories: Dict[int, List[np.ndarray]] = {q: [] for q in selected_qubits}
    entanglement_histories: Dict[int, List[float]] = {q: [] for q in selected_qubits}
    gate_labels: List[str] = []
    
    for q in selected_qubits:
        histories[q].append(bloch_vector(state, q))
        entanglement_histories[q].append(calculate_entanglement(state, q))
    gate_labels.append("Initial |0⟩")
    
    for idx, instruction in enumerate(circuit.data, start=1):
        if instruction.operation.name.lower() == "barrier":
            continue
        
        qubit_indices = [circuit.find_bit(qb).index for qb in instruction.qubits]
        
        state = state.evolve(instruction.operation, qargs=qubit_indices)
        gate_labels.append(f"Gate {idx}: {instruction.operation.name.upper()}")
        
        for q in selected_qubits:
            histories[q].append(bloch_vector(state, q))
            entanglement_histories[q].append(calculate_entanglement(state, q))
    
    if len(gate_labels) <= 1:
        raise ValueError("Circuit contains no gates to animate.")
    
    trajectories = {q: np.vstack(vectors) for q, vectors in histories.items()}
    entanglement_values = {q: np.array(values) for q, values in entanglement_histories.items()}
    return trajectories, entanglement_values, gate_labels


def _ultra_sparse_approximation_no_aer(circuit: QuantumCircuit, selected_qubits: List[int], 
                                       sample_points: List[int], total_gates: int) -> tuple:
    print("⚠️ Using approximate visualization (not exact simulation)")
    
    histories: Dict[int, List[np.ndarray]] = {q: [] for q in selected_qubits}
    ent_histories: Dict[int, List[float]] = {q: [] for q in selected_qubits}
    gate_labels: List[str] = ["Initial |0⟩ (approx)"]
    
    for q in selected_qubits:
        histories[q].append(np.array([0.0, 0.0, 1.0])) 
        ent_histories[q].append(0.0)
    
    np.random.seed(42) 
    for i, sp in enumerate(sample_points[1:], 1):
        gate_labels.append(f"Gate ~{sp}/{total_gates} (approx)")
        progress = i / len(sample_points)
        
        for q in selected_qubits:
            # Gradually randomize position
            noise_scale = 0.3 * progress
            prev = histories[q][-1]
            new_vec = prev + np.random.randn(3) * noise_scale
            # Normalize to stay on Bloch sphere
            norm = np.linalg.norm(new_vec)
            if norm > 0:
                new_vec = new_vec / norm
            else:
                new_vec = np.array([0.0, 0.0, 1.0])
            
            histories[q].append(new_vec)
            # Approximate entanglement increases over time in typical circuits
            ent_histories[q].append(min(0.95, 0.2 * progress + 0.1 * np.random.rand()))
    
    trajectories = {q: np.vstack(v) for q, v in histories.items()}
    entanglement_values = {q: np.array(v) for q, v in ent_histories.items()}
    return trajectories, entanglement_values, gate_labels


def circuit_bloch_trajectories_ultra_sparse(circuit: QuantumCircuit, selected_qubits: List[int] = None) -> tuple:
    if not selected_qubits:
        selected_qubits = list(range(min(3, circuit.num_qubits)))
    
    logical_gates = [inst for inst in circuit.data if inst.operation.name.lower() != 'barrier']
    total_gates = len(logical_gates)
    if total_gates == 0:
        raise ValueError("Circuit contains no gates to animate.")
    
    # Only 5-10 snapshots for ultra-large circuits
    num_samples = min(8, max(5, total_gates // 50))
    sample_interval = max(1, total_gates // num_samples)
    sample_points = [0] + list(range(sample_interval, total_gates, sample_interval))
    if sample_points[-1] != total_gates - 1:
        sample_points.append(total_gates - 1)
    
    print(f"⚡ Ultra-sparse: {len(sample_points)} snapshots from {total_gates} gates")
    
    # Use MPS simulation to avoid memory issues
    if not AER_AVAILABLE:
        print("⚠️ Aer not available - using classical approximation (random states)")
        return _ultra_sparse_approximation_no_aer(circuit, selected_qubits, sample_points, total_gates)
    
    histories: Dict[int, List[np.ndarray]] = {q: [] for q in selected_qubits}
    ent_histories: Dict[int, List[float]] = {q: [] for q in selected_qubits}
    gate_labels: List[str] = []
    
    sim = AerSimulator(method='matrix_product_state')
    
    # Helper to build prefix circuit
    def build_prefix(orig: QuantumCircuit, upto_gate_exclusive: int) -> QuantumCircuit:
        qc = QuantumCircuit(orig.num_qubits)
        if upto_gate_exclusive == 0:
            return qc
        non_barrier_seen = 0
        for inst in orig.data:
            if inst.operation.name.lower() == 'barrier':
                continue
            if non_barrier_seen >= upto_gate_exclusive:
                break
            qc.append(inst.operation, [circuit.find_bit(qb).index for qb in inst.qubits])
            non_barrier_seen += 1
        return qc
    
    # Initial state - for |0⟩ we know the Bloch vectors analytically
    gate_labels.append("Initial |0⟩")
    for q in selected_qubits:
        histories[q].append(np.array([0.0, 0.0, 1.0]))  # |0⟩ is at north pole
        ent_histories[q].append(0.0)  # No entanglement initially
    
    # Sample at key points - use save_density_matrix for individual qubits
    for sp in sample_points[1:]:  # Skip 0 since we already did initial state
        prefix_qc = build_prefix(circuit, sp + 1)
        
        gate_labels.append(f"Gate ~{sp}/{total_gates}")
        
        # For each qubit, save its reduced density matrix
        for q in selected_qubits:
            # Create circuit that saves density matrix for just this qubit
            qc_temp = prefix_qc.copy()
            qc_temp.save_density_matrix([q])
            
            try:
                result = sim.run(qc_temp, shots=1).result()
                rho = result.data()['density_matrix']
                
                # Calculate Bloch vector from density matrix
                rx = float(np.real(np.trace(rho @ PAULI_X)))
                ry = float(np.real(np.trace(rho @ PAULI_Y)))
                rz = float(np.real(np.trace(rho @ PAULI_Z)))
                histories[q].append(np.array([rx, ry, rz]))
                
                # Calculate entanglement from purity
                purity = float(np.real(np.trace(rho @ rho)))
                # Von Neumann entropy approximation from purity
                if purity > 0.99:
                    entropy = 0.0
                else:
                    entropy = -purity * np.log2(purity + 1e-10) - (1 - purity) * np.log2(1 - purity + 1e-10)
                ent_histories[q].append(min(1.0, entropy))
                
            except Exception as e:
                print(f"⚠️ Error computing qubit {q} at step {sp}: {e}")
                # Use previous value or default
                if len(histories[q]) > 0:
                    histories[q].append(histories[q][-1])
                    ent_histories[q].append(ent_histories[q][-1])
                else:
                    histories[q].append(np.array([0.0, 0.0, 1.0]))
                    ent_histories[q].append(0.0)
    
    trajectories = {q: np.vstack(v) for q, v in histories.items()}
    entanglement_values = {q: np.array(v) for q, v in ent_histories.items()}
    return trajectories, entanglement_values, gate_labels


def circuit_bloch_trajectories_mps(circuit: QuantumCircuit, selected_qubits: List[int] = None) -> tuple:

    if not AER_AVAILABLE:
        print("⚠️ AerSimulator not available; falling back to exact/sparse path.")
        return circuit_bloch_trajectories(circuit, selected_qubits)

    if not selected_qubits:
        selected_qubits = list(range(min(3, circuit.num_qubits)))

    logical_gates = [inst for inst in circuit.data if inst.operation.name.lower() != 'barrier']
    total_gates = len(logical_gates)
    if total_gates == 0:
        raise ValueError("Circuit contains no gates to animate.")

    sample_interval = max(1, total_gates // 20)  
    sample_points = list(range(0, total_gates, sample_interval))
    if sample_points[-1] != total_gates - 1:
        sample_points.append(total_gates - 1)

    histories: Dict[int, List[np.ndarray]] = {q: [] for q in selected_qubits}
    ent_histories: Dict[int, List[float]] = {q: [] for q in selected_qubits}
    gate_labels: List[str] = ["Initial |0⟩"]

    sim = AerSimulator(method='matrix_product_state')

    # Build prefix helper
    def build_prefix(orig: QuantumCircuit, upto_gate_exclusive: int) -> QuantumCircuit:
        qc = QuantumCircuit(orig.num_qubits)
        if upto_gate_exclusive == 0:
            return qc
        non_barrier_seen = 0
        for inst in orig.data:
            if inst.operation.name.lower() == 'barrier':
                continue
            if non_barrier_seen >= upto_gate_exclusive:
                break
            qargs = [qc.qubits[orig.find_bit(qb).index] for qb in inst.qubits]
            qc.append(inst.operation, qargs)
            non_barrier_seen += 1
        return qc

    # Initial state snapshot
    initial_state = Statevector.from_label("0" * circuit.num_qubits)
    for q in selected_qubits:
        histories[q].append(bloch_vector(initial_state, q))
        ent_histories[q].append(calculate_entanglement(initial_state, q))

    for sp in sample_points:
        prefix_qc = build_prefix(circuit, sp + 1)  # include gate at sp
        try:
            result = sim.run(prefix_qc).result()
            sv = result.get_statevector(prefix_qc, decimals=0)
            state = Statevector(sv)
        except Exception as e:
            print(f"⚠️ Aer MPS failure at gate {sp}: {e}; falling back to manual evolve for this snapshot.")
            state = Statevector.from_label("0" * circuit.num_qubits)
            non_barrier_seen = 0
            for inst in circuit.data:
                if inst.operation.name.lower() == 'barrier':
                    continue
                qargs = [circuit.find_bit(qb).index for qb in inst.qubits]
                state = state.evolve(inst.operation, qargs=qargs)
                if non_barrier_seen == sp:
                    break
                non_barrier_seen += 1

        # Record snapshot
        op_name = logical_gates[sp].operation.name.upper()
        gate_labels.append(f"Gate {sp+1}: {op_name}")
        for q in selected_qubits:
            histories[q].append(bloch_vector(state, q))
            ent_histories[q].append(calculate_entanglement(state, q))

    trajectories = {q: np.vstack(v) for q, v in histories.items()}
    entanglement_values = {q: np.array(v) for q, v in ent_histories.items()}
    return trajectories, entanglement_values, gate_labels


def render_circuit_to_surface(circuit: QuantumCircuit, width: int, height: int) -> pygame.Surface:
    """Render quantum circuit diagram using Matplotlib and convert to Pygame surface."""
    # Create matplotlib figure
    fig = circuit.draw('mpl', style={'backgroundcolor': '#1a1a2e'})
    
    # Render to buffer
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Convert to PIL Image
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    img = Image.frombytes('RGBA', (w, h), buf)
    
    # Convert PIL to Pygame surface
    mode = img.mode
    size = img.size
    data = img.tobytes()
    py_img = pygame.image.frombytes(data, size, mode)
    
    # Scale to desired size
    py_img = pygame.transform.smoothscale(py_img, (width, height))
    
    plt.close(fig)
    return py_img


class BlochSphere:
    
    def __init__(self, center_x: int, center_y: int, radius: int):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.rotation_x = 20.0  # Tilt angle for 3D effect (in degrees)
        self.rotation_y = 45.0  # Rotation around Y axis (azimuth)
        
        # Mouse interaction state
        self.dragging = False
        self.last_mouse_pos = None
        
    def handle_mouse(self, event, mouse_pos) -> bool:
        # Check if mouse is in this sphere's area
        mx, my = mouse_pos
        dx = mx - self.center_x
        dy = my - self.center_y
        distance = math.sqrt(dx*dx + dy*dy)
        in_bounds = distance <= self.radius * 1.5
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if in_bounds:
                self.dragging = True
                self.last_mouse_pos = mouse_pos
                return True
                
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                self.last_mouse_pos = None
                return True
            
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            if self.last_mouse_pos:
                dx = mouse_pos[0] - self.last_mouse_pos[0]
                dy = mouse_pos[1] - self.last_mouse_pos[1]
                
                # Update rotation based on mouse movement
                self.rotation_y += dx * 0.5
                self.rotation_x -= dy * 0.5
                
                self.rotation_x = max(-89, min(89, self.rotation_x))
                
                self.last_mouse_pos = mouse_pos
                return True
                
        return False
    
    def project_3d_to_2d(self, x: float, y: float, z: float) -> tuple:
        # Convert angles to radians
        rot_x_rad = math.radians(self.rotation_x)
        rot_y_rad = math.radians(self.rotation_y)
        
        # Rotate around Y axis (azimuth)
        x_rot = x * math.cos(rot_y_rad) + z * math.sin(rot_y_rad)
        z_temp = -x * math.sin(rot_y_rad) + z * math.cos(rot_y_rad)
        
        # Rotate around X axis (elevation)
        y_rot = y * math.cos(rot_x_rad) - z_temp * math.sin(rot_x_rad)
        z_rot = y * math.sin(rot_x_rad) + z_temp * math.cos(rot_x_rad)
        
        # Project to 2D
        screen_x = self.center_x + int(x_rot * self.radius)
        screen_y = self.center_y - int(y_rot * self.radius)
        
        return screen_x, screen_y, z_rot  # Return depth for sorting
    
    def draw(self, surface: pygame.Surface, bloch_vec: np.ndarray, entanglement: float = 0.0):
        # Draw entanglement-colored background circle
        # More entanglement = more blue
        entanglement_color = self._get_entanglement_color(entanglement)
        pygame.draw.circle(surface, entanglement_color, (self.center_x, self.center_y), 
                         self.radius, 0)  # Filled circle
        
        # Draw highlight circle if being dragged
        if self.dragging:
            pygame.draw.circle(surface, (100, 200, 255), (self.center_x, self.center_y), 
                             int(self.radius * 1.5), 3)
        
        # Draw wireframe circles
        self._draw_wireframe(surface)
        
        # Draw coordinate axes
        self._draw_axes(surface)
        
        # Draw state vector
        if bloch_vec is not None:
            self._draw_state_vector(surface, bloch_vec)
    
    def _get_entanglement_color(self, entanglement: float) -> tuple:
        """Convert entanglement value (0-1) to color (white to blue)."""
        # 0 entanglement = white (248, 248, 250)
        # 1 entanglement = deep blue (50, 100, 200)
        white = np.array([248, 248, 250])
        blue = np.array([50, 100, 200])
        
        color = white + entanglement * (blue - white)
        return tuple(int(c) for c in color)
    
    def _draw_wireframe(self, surface: pygame.Surface):
        """Draw sphere wireframe - simple circle with equator for depth."""
        wireframe_color = (150, 150, 150)
        
        # Draw outer circle boundary
        pygame.draw.circle(surface, wireframe_color, (self.center_x, self.center_y), self.radius, 3)
        
        # Draw equator (horizontal circle in 3D) for depth perception
        equator_points = []
        for i in range(40):
            angle = 2 * np.pi * i / 40
            x = np.cos(angle)
            y = np.sin(angle)
            z = 0
            screen_x, screen_y, depth = self.project_3d_to_2d(x, y, z)
            equator_points.append((screen_x, screen_y))
        if len(equator_points) > 1:
            pygame.draw.lines(surface, (100, 100, 100), True, equator_points, 2)
    
    def _draw_axes(self, surface: pygame.Surface):
        """Draw X, Y, Z axes with labels."""
        font = pygame.font.Font(None, 24)
        
        # Simple blue color for all axes
        axis_color = (80, 120, 200)
        label_color = (60, 60, 60)
        
        # X axis
        x1, y1, _ = self.project_3d_to_2d(-1.2, 0, 0)
        x2, y2, _ = self.project_3d_to_2d(1.2, 0, 0)
        pygame.draw.line(surface, axis_color, (x1, y1), (x2, y2), 3)
        label = font.render("X", True, label_color)
        surface.blit(label, (x2 + 5, y2 - 10))
        
        # Y axis
        x1, y1, _ = self.project_3d_to_2d(0, -1.2, 0)
        x2, y2, _ = self.project_3d_to_2d(0, 1.2, 0)
        pygame.draw.line(surface, axis_color, (x1, y1), (x2, y2), 3)
        label = font.render("Y", True, label_color)
        surface.blit(label, (x2 + 5, y2 - 10))
        
        # Z axis
        x1, y1, _ = self.project_3d_to_2d(0, 0, -1.2)
        x2, y2, _ = self.project_3d_to_2d(0, 0, 1.2)
        pygame.draw.line(surface, axis_color, (x1, y1), (x2, y2), 3)
        
        # |0⟩ at north pole
        label_0 = font.render("|0⟩", True, label_color)
        surface.blit(label_0, (x2 - 15, y2 - 25))
        
        # |1⟩ at south pole
        label_1 = font.render("|1⟩", True, label_color)
        surface.blit(label_1, (x1 - 15, y1 + 10))
    
    def _draw_trail(self, surface: pygame.Surface, trail: List[np.ndarray]):
        """Draw the path traced by the state vector."""
        if len(trail) < 2:
            return
        
        trail_points = []
        for vec in trail:
            if len(vec) >= 3:
                x, y, z = vec[0], vec[1], vec[2]
                screen_x, screen_y, _ = self.project_3d_to_2d(x * 0.95, y * 0.95, z * 0.95)
                trail_points.append((screen_x, screen_y))
        
        if len(trail_points) > 1:
            # Draw trail with gradient effect
            for i in range(len(trail_points) - 1):
                alpha = int(255 * (i + 1) / len(trail_points))
                color = (70, 200, 255)
                thickness = 2 if i > len(trail_points) - 5 else 1
                pygame.draw.line(surface, color, trail_points[i], trail_points[i + 1], thickness)
    
    def _draw_state_vector(self, surface: pygame.Surface, bloch_vec: np.ndarray):
        """Draw the state vector arrow and point."""
        x, y, z = bloch_vec[0], bloch_vec[1], bloch_vec[2]
        
        # Draw vector line from center
        cx, cy, _ = self.project_3d_to_2d(0, 0, 0)
        sx, sy, depth = self.project_3d_to_2d(x * 0.95, y * 0.95, z * 0.95)
        
        # Draw thick red arrow line
        pygame.draw.line(surface, (255, 50, 50), (cx, cy), (sx, sy), 5)
        
        # Draw arrowhead at the end
        pygame.draw.circle(surface, (255, 50, 50), (sx, sy), 8)


class PygameQuantumVisualizer:
    """Main visualizer combining circuit diagram and Bloch sphere animation."""
    
    def __init__(self, width: int = 1400, height: int = 900):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Quantum Circuit Bloch Visualizer")
        self.clock = pygame.time.Clock()
        
        # Font setup
        self.title_font = pygame.font.Font(None, 48)
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # Circuit files – restore all 1–6; large ones will be displayed but heavy simulation may be disabled
        self.circuit_files = [
            'circuits/1.qasm', 'circuits/2.qasm', 'circuits/3.qasm',
            'circuits/4.qasm', 'circuits/5.qasm', 'circuits/6.qasm'
        ]
        self.current_circuit_idx = 0
        
        # Animation state
        self.is_playing = False
        self.current_frame = 0
        self.animation_speed = 30  # frames per gate step
        self.frame_counter = 0
        
        # Circuit data
        self.circuit = None
        self.trajectories = None
        self.entanglement = None
        self.gate_labels = None
        self.circuit_surface = None
        self.num_qubits = 0
        self.simulation_disabled = False  # Flag to indicate too-large circuit for simulation
        self.simulation_mode = 'exact'
        
        # Bloch spheres (we'll show up to 3 qubits)
        self.bloch_spheres = []
        self.visible_qubits = []
        
        # Load first circuit
        self.load_circuit(self.circuit_files[self.current_circuit_idx])
    
    def load_circuit(self, filepath: str):
        """Load a new quantum circuit."""
        try:
            full_path = f"/Users/georgeliao/Desktop/hackathon/Hackathon 2/2025_Hackathon_Challenge/2025_Hackathon_Challenge/{filepath}"
            self.circuit = load_circuit(full_path)
            self.num_qubits = self.circuit.num_qubits

            # Sanity check: if parsing produced an empty circuit, skip and try next
            if self.num_qubits <= 0:
                raise ValueError("Parsed empty circuit (0 qubits)")

            # For extremely large circuits (>20 qubits), use ultra-sparse approximation
            if self.num_qubits > 20:
                print(f"⚠️ Circuit has {self.num_qubits} qubits (>20).")
                print(f"📉 Will use ultra-sparse approximation (5-10 snapshots only)")
                self.simulation_mode = 'ultra-sparse'
                self.simulation_disabled = False

            # Do not render circuit diagram (user requested to leave it out to save space)
            self.circuit_surface = None
            
            # Setup Bloch spheres - use default qubits (no user input during circuit switching)
            # For circuits with many qubits, show first 3 qubits by default
            self.visible_qubits = list(range(min(3, self.num_qubits)))
            
            if not self.simulation_disabled and self.num_qubits > 3:
                print(f"\n📊 Circuit has {self.num_qubits} qubits. Displaying first 3: {self.visible_qubits}")
                print(f"💡 You can modify visible_qubits in the code to show different qubits")
            
            # Check if circuit is too large for exact simulation
            if self.num_qubits > 8:
                print(f"⚠️ Circuit has {self.num_qubits} qubits (> 8).")
                print(f"💡 Computing trajectories for selected qubits only: {self.visible_qubits}")
            
            if not self.simulation_disabled:
                # Choose simulation strategy based on circuit size
                if self.num_qubits > 20:
                    # Ultra-sparse for massive circuits
                    print(f"⏳ Computing Bloch trajectories (ultra-sparse) for qubits {self.visible_qubits}...")
                    self.simulation_mode = 'ultra-sparse'
                    self.trajectories, self.entanglement, self.gate_labels = circuit_bloch_trajectories_ultra_sparse(self.circuit, self.visible_qubits)
                elif AER_AVAILABLE and self.num_qubits > 8:
                    # MPS for medium-large circuits
                    print(f"⏳ Computing Bloch trajectories (MPS) for qubits {self.visible_qubits}...")
                    self.simulation_mode = 'MPS'
                    self.trajectories, self.entanglement, self.gate_labels = circuit_bloch_trajectories_mps(self.circuit, self.visible_qubits)
                else:
                    # Exact or sparse for smaller circuits
                    print(f"⏳ Computing Bloch trajectories (exact/sparse) for qubits {self.visible_qubits}...")
                    self.simulation_mode = 'exact' if self.num_qubits <= 8 else 'sparse'
                    self.trajectories, self.entanglement, self.gate_labels = circuit_bloch_trajectories(self.circuit, self.visible_qubits)
            
            self.setup_bloch_spheres()
            
            # Reset animation
            self.current_frame = 0
            self.frame_counter = 0
            self.is_playing = False
            
            if self.simulation_disabled:
                print(f"✅ Loaded circuit {self.current_circuit_idx + 1}: {self.num_qubits} qubits (simulation disabled)")
            else:
                print(f"✅ Loaded circuit {self.current_circuit_idx + 1}: {self.num_qubits} qubits, {len(self.gate_labels)} steps")
                print(f"✅ Visualizing qubits: {self.visible_qubits}")
            
        except Exception as e:
            print(f"❌ Error loading circuit: {e}")
            # Try next circuit
            self.current_circuit_idx = (self.current_circuit_idx + 1) % len(self.circuit_files)
            if self.current_circuit_idx != 0:  # Avoid infinite loop
                self.load_circuit(self.circuit_files[self.current_circuit_idx])
    
    def setup_bloch_spheres(self):
        """Initialize Bloch sphere renderers."""
        self.bloch_spheres = []
        sphere_y = 680
        sphere_spacing = 350
        # Start at 460px to be well within the white panel (which starts at 260px)
        sphere_start_x = 460
        
        for i, qubit_idx in enumerate(self.visible_qubits):
            center_x = sphere_start_x + i * sphere_spacing
            self.bloch_spheres.append(BlochSphere(center_x, sphere_y, 120))
    
    def update(self):
        """Update animation state."""
        if self.is_playing and self.trajectories:
            self.frame_counter += 1
            
            if self.frame_counter >= self.animation_speed:
                self.frame_counter = 0
                self.current_frame += 1
                
                # Loop animation
                max_frames = len(self.gate_labels) - 1
                if self.current_frame > max_frames:
                    self.current_frame = 0
    
    def draw(self):
        """Render everything to screen."""
        # Background
        self.screen.fill((15, 15, 30))
        
        # Title
        title = self.title_font.render(
            f"Quantum Circuit Visualizer - Circuit {self.current_circuit_idx + 1}/{len(self.circuit_files)}",
            True, (255, 255, 255)
        )
        self.screen.blit(title, (50, 20))
        
        # Circuit diagram
        # (Disabled) Circuit diagram – omitted per user's request
        
        # Gate label or simulation-disabled notice
        if self.simulation_disabled:
            notice = self.font.render(
                f"Circuit {self.current_circuit_idx + 1}: {self.num_qubits} qubits – simulation disabled (too large)",
                True, (255, 180, 120)
            )
            self.screen.blit(notice, (50, 450))
        elif self.gate_labels and self.current_frame < len(self.gate_labels):
            gate_text = self.font.render(
                f"Step {self.current_frame + 1}/{len(self.gate_labels)}: {self.gate_labels[self.current_frame]}",
                True, (100, 255, 150) if self.is_playing else (255, 200, 100)
            )
            self.screen.blit(gate_text, (50, 450))
        # Mode indicator
        mode_text = self.small_font.render(f"Mode: {self.simulation_mode}", True, (180,180,220))
        self.screen.blit(mode_text, (50, 420))
        
        # Status
        status = "▶ Playing" if self.is_playing else "⏸ Paused"
        status_text = self.font.render(status, True, (100, 255, 150) if self.is_playing else (255, 200, 100))
        self.screen.blit(status_text, (50, 500))
        
        # White panel for Bloch spheres (separate from text pane)
        bloch_panel_y = 530
        bloch_panel_height = self.height - bloch_panel_y - 20
        # Leave space on the left for instructions (250px)
        bloch_panel = pygame.Rect(260, bloch_panel_y, self.width - 270, bloch_panel_height)
        
        # Draw white/off-white background
        white_color = (248, 248, 250)
        pygame.draw.rect(self.screen, white_color, bloch_panel)
        pygame.draw.rect(self.screen, (200, 200, 200), bloch_panel, 3)
        
        # Draw Bloch spheres on the white panel
        if not self.simulation_disabled and self.trajectories and self.entanglement:
            for i, qubit_idx in enumerate(self.visible_qubits):
                if qubit_idx in self.trajectories:
                    # Get current state (discrete state at each gate step)
                    trajectory = self.trajectories[qubit_idx]
                    current_vec = trajectory[self.current_frame] if self.current_frame < len(trajectory) else trajectory[-1]
                    
                    # Get current entanglement value
                    entanglement_array = self.entanglement[qubit_idx]
                    current_entanglement = entanglement_array[self.current_frame] if self.current_frame < len(entanglement_array) else entanglement_array[-1]
                    
                    # Debug: Print entanglement values
                    if self.is_playing and qubit_idx == self.visible_qubits[0]:  # Only print for first visible qubit
                        print(f"Frame {self.current_frame}: Qubit {qubit_idx} entanglement = {current_entanglement:.3f}")
                    
                    # Draw sphere with entanglement coloring
                    self.bloch_spheres[i].draw(self.screen, current_vec, current_entanglement)
                    
                    # Qubit label (black text on white background)
                    label = self.font.render(f"Qubit {qubit_idx}", True, (20, 20, 20))
                    label_rect = label.get_rect(center=(self.bloch_spheres[i].center_x, 
                                                         self.bloch_spheres[i].center_y + 160))
                    self.screen.blit(label, label_rect)
                    
                    # State coordinates (black text on white background)
                    coord_text = f"({current_vec[0]:.2f}, {current_vec[1]:.2f}, {current_vec[2]:.2f})"
                    coord_label = self.small_font.render(coord_text, True, (40, 40, 40))
                    coord_rect = coord_label.get_rect(center=(self.bloch_spheres[i].center_x,
                                                               self.bloch_spheres[i].center_y + 185))
                    self.screen.blit(coord_label, coord_rect)
                    
                    # Entanglement value
                    ent_text = f"Entanglement: {current_entanglement:.3f}"
                    ent_label = self.small_font.render(ent_text, True, (50, 100, 200))
                    ent_rect = ent_label.get_rect(center=(self.bloch_spheres[i].center_x,
                                                           self.bloch_spheres[i].center_y + 210))
                    self.screen.blit(ent_label, ent_rect)
        
        # Instructions
        instructions = [
            "🎮 CONTROLS:",
            "SPACE = Play/Pause",
            "LEFT/RIGHT = Change Circuit",
            "R = Reset Animation",
            "UP/DOWN = Speed",
            "DRAG MOUSE = Rotate Bloch Spheres",
            "Q = Quit"
        ]
        
        y_offset = self.height - 220
        for instruction in instructions:
            color = (255, 255, 150) if instruction.startswith("🎮") else (180, 180, 200)
            text = self.small_font.render(instruction, True, color)
            self.screen.blit(text, (50, y_offset))
            y_offset += 25
    
    def handle_event(self, event):
        """Handle user input."""
        # Let Bloch spheres handle mouse events first
        mouse_pos = pygame.mouse.get_pos()
        for sphere in self.bloch_spheres:
            if sphere.handle_mouse(event, mouse_pos):
                return True  # Event was handled by a Bloch sphere
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.is_playing = not self.is_playing
                print(f"{'▶ Playing' if self.is_playing else '⏸ Paused'}")
            
            elif event.key == pygame.K_r:
                self.current_frame = 0
                self.frame_counter = 0
                self.is_playing = False
                print("🔄 Animation reset")
            
            elif event.key == pygame.K_LEFT:
                self.current_circuit_idx = (self.current_circuit_idx - 1) % len(self.circuit_files)
                self.load_circuit(self.circuit_files[self.current_circuit_idx])
                print(f"⬅ Circuit {self.current_circuit_idx + 1}")
            
            elif event.key == pygame.K_RIGHT:
                self.current_circuit_idx = (self.current_circuit_idx + 1) % len(self.circuit_files)
                self.load_circuit(self.circuit_files[self.current_circuit_idx])
                print(f"➡ Circuit {self.current_circuit_idx + 1}")
            
            elif event.key == pygame.K_UP:
                self.animation_speed = max(5, self.animation_speed - 5)
                print(f"⚡ Speed: {self.animation_speed}")
            
            elif event.key == pygame.K_DOWN:
                self.animation_speed = min(60, self.animation_speed + 5)
                print(f"🐌 Speed: {self.animation_speed}")
            
            elif event.key == pygame.K_q:
                return False
        
        return True
    
    def run(self):
        """Main game loop."""
        print("\n🎮 Pygame Quantum Visualizer Started!")
        print("=" * 60)
        print("✨ Features:")
        print("  • High-quality circuit diagrams from Qiskit")
        print("  • Smooth Bloch sphere animations")
        print("  • Interactive controls")
        print("=" * 60)
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    running = self.handle_event(event)
            
            self.update()
            self.draw()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        print("\n👋 Visualizer closed")


if __name__ == "__main__":
    visualizer = PygameQuantumVisualizer()
    visualizer.run()
