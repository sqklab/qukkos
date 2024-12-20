namespace QUKKOS 
{
open Microsoft.Quantum.Intrinsic;
// Deuteron ansatz (to be used a QUKKOS NISQ kernel)
operation ansatz(qubits : Qubit[], theta : Double) : Unit {
  X(qubits[0]);
  Ry(theta, qubits[1]);
  CNOT(qubits[1], qubits[0]);          
}
}