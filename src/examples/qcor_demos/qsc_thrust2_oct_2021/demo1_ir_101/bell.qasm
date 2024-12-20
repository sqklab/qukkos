// qukkos bell.qasm -o bell.x
// ./bell.x 
//
// qukkos -v bell.qasm -o bell.x
// qukkos --emit-mlir bell.qasm
// qukkos --emit-llvm bell.qasm

OPENQASM 3;

qubit q[2];

const shots = 15;
int count = 0;

for i in [0:15] {
    
    // Create Bell state
    h q[0];
    cnot q[0], q[1];
    
    // Measure and assert both are equal
    bit c[2];
    c = measure q;
    if (c[0] == c[1]) {
        print("iter", i, ": measured =", c[0], c[1]);
        count += 1;
    }

    reset q;
}

print("count is", count);

