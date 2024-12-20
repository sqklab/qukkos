# Using QUKKOS FTQC runtime, which supports 
# fast feedforward of measurement results.

from qukkos import qjit, qalloc, qreg

# Encode qubits into logical qubits:
# Assume q[0], q[3], q[6] are initial physical qubits
# that will be mapped to logical qubits q[0-2], q[3-5], etc.
@qjit
def encodeLogicalQubit(q : qreg):
    nbLogicalQubits = q.size() / 3
    for i in range(nbLogicalQubits):
        physicalQubitIdx = 3 * i
        CX(q[physicalQubitIdx], q[physicalQubitIdx + 1])
        CX(q[physicalQubitIdx], q[physicalQubitIdx + 2])

# Measure syndromes of a logical qubit using the provided ancilla
# Assume that the ancilla is in |0> state and must be returned in that state.
@qjit
def measureSyndrome(q : qreg, logicalIdx: int, ancIdx: int):
    physicalIdx = logicalIdx * 3
    CX(q[physicalIdx], q[ancIdx])
    CX(q[physicalIdx + 1], q[ancIdx])
    parity01 = Measure(q[ancIdx])
    if parity01: 
        #Reset anc qubit
        X(q[ancIdx])
    
    CX(q[physicalIdx + 1], q[ancIdx])
    CX(q[physicalIdx + 2], q[ancIdx])
    parity12 = Measure(q[ancIdx])
    if parity12:
        #Reset anc qubit
        X(q[ancIdx])
    # Note: only in FTQC runtime that we can examine the measure results in realtime.
    print("Parity01 =", parity01, "Parity12 =", parity12)

@qjit
def testBitflipCode(q : qreg):
    H(q[0])
    encodeLogicalQubit(q)
    measureSyndrome(q, 0, 3)
    # Apply an X error
    for i in range(3):
        print("Apply X error @ ", i)
        X(q[i])
        measureSyndrome(q, 0, 3)
        # Cancel the error for the next test
        X(q[i])


# Allocate 4 qubits: 3 qubits + 1 ancilla
q = qalloc(4)

testBitflipCode(q)