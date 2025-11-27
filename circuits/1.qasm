# Taken from QuEra's repository: https://github.com/iQuHACK/2025-QuEra/blob/main/iQuHack-2025.pdf
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];

cz q[0],q[1];
cx q[2],q[1];