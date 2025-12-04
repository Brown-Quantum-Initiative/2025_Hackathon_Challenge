# Taken from QuEra's repository: https://github.com/iQuHACK/2025-QuEra/blob/main/iQuHack-2025.pdf
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
ccx q[0],q[1],q[2];