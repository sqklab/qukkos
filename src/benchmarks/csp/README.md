Scripts and results of benchmarking QUKKOS's circuit optimization.

- Main script `driver.py`: execute QUKKOS compilation and collect optimization data.

- Test suites: sub-folders of the `resources` directory.
Note: we only kept a minimal subset of those QASM sources here for demonstration purposes.

- Result `.csv` files collected from running the extended test suites which are available at this [repo](https://github.com/tnguyen-ornl/qukkos/tree/tnguyen/opt-data/benchmarks/resources).

Note: This benchmarking script also uses an external XACC plugin for VOQC circuit optimization. The plugin can be installed from [here](https://github.com/tnguyen-ornl/SQIR).
