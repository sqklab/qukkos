![qukkos](docs/assets/qukkos_full_logo.svg)

| master | 


# QUKKOS

QUKKOS is a C++ language extension and associated compiler implementation
for hybrid quantum-classical programming. Transparent device selection on two different heterogeneous systems.

## Documentation

* [Documentation and User Guides]https://github.com/sqkcloud/deploy)
* [Doxygen API Docs](https://github.com/sqkcloud/docs)

## Installation
To install the `qukkos` nightly binaries (for Mac OS X and Linux x86_64) run the following command from your terminal 
```bash
/bin/bash -c "$(curl -fsSL https://github.com/sqkcloud/docs/install.sh)"
```
To use the Python API, be sure to set your `PYTHONPATH`. 
For more details, see the [full installation documentation page](https://github.com/sqkcloud/docs/getting_started/).

### Docker Images

Nightly docker images are also available that serve up a [VSCode IDE](https://github.com/cdr/code-server) on port 8080. To use this image, run 
```bash
docker run -it -p 8080:8080 qukkos/qukkos
```
and navigate to ``https://localhost:8080`` in your browser to open the IDE and get started with QUKKOS. 

Alternatively, you could use the `qukkos/cli` image providing simple command-line access to the `qukkos` compiler. 
```bash
docker run -it qukkos/cli
```

## Cite QUKKOS 
If you use qukkos in your research, please use the following citation 
```
```

## Feedback

If you have feedback about the content in this repository, please let us know by
filing a [new issue](https://github.com/sqkcloud/qukkos/issues/new)!

## Contributing

There are many ways in which you can contribute to QUKKOS, whether by contributing
a feature or by engaging in discussions; we value contributions in all shapes
and sizes! We refer to [this document](CONTRIBUTING.md) for guidelines and ideas
for how you can get involved.

Contributing a pull request to this repo requires to agree to a
[Contributor License Agreement (CLA)](https://en.wikipedia.org/wiki/Contributor_License_Agreement)
declaring that you have the right to, and actually do, grant us the rights to
use your contribution. We are still working on setting up a suitable CLA-bot to
automate this process. A CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately. Once it is set up, simply
follow the instructions provided by the bot. You will only need to do this once.

## Code of Conduct

This project has adopted the community covenant
[Code of Conduct](https://github.com/sqkcloud/.github/blob/main/Code_of_Conduct.md#contributor-covenant-code-of-conduct).
Please contact [sqkadmin@sqkcloud.com](mailto:sqkadmin@sqkcloud.com) for Code of
Conduct issues or inquires.
