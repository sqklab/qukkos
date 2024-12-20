// NISQ Mode Execution
//
// Compile and run on qpp 
// qukkos ghz.qasm -o ghz.x -qrt nisq -shots 1000
// ./ghz.x 
//
// Show on local aer with noisy backend
// qukkos ghz.qasm -o ghz.x -qrt nisq -shots 100 -qpu aer:ibmq_sydney
// ./ghz.x 
//
// (Now Show off how this works, MLIR + QIR)
//
// qukkos -v ghz.qasm -o ghz.x
// qukkos --emit-mlir ghz.qasm
// qukkos --emit-llvm ghz.qasm


OPENQASM 3;

const n_qubits = 3;

qubit q[n_qubits];

gate ctrl_x a, b {
    ctrl @ x a, b;
}

h q[0];
for i in [0:n_qubits-1] {
    ctrl_x q[i], q[i+1];
}

bit c[n_qubits];
c = measure q;