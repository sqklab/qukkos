__qpu__ void bernstein_vazirani(qreg q, std::string &secret_bits) {
    // prepare ancilla in |1>
    X(q[secret_bits.size()]);

    // input superpositions
    for (int i = 0; i <= secret_bits.size(); i++) {
        H(q[i]);
    }

    // oracle
    for (int i = 0; i <= secret_bits.size(); i++) {
        if (i == secret_bits.size()) {
            // Silly condition just to test continue keyword
            continue;
        }

        if (secret_bits[i] == '1') {
            CX(q[i], q[secret_bits.size()]);
        }
    }

    H(q);

    // Use this instead of head() to test that curly braces work
    Measure(q.extract_range({0, secret_bits.size()}));
}

int main() {
    set_shots(1024);
    std::string secret_bits = "110101";

    auto q = qalloc(secret_bits.size() + 1);
    bernstein_vazirani(q, secret_bits);
    q.print();

    qukkos_expect(q.counts().size() == 1);
    qukkos_expect(q.counts()[secret_bits] == 1024);
}
