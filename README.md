# Unified Collective Communications (UCC)


![UCC Logo](docs/images/ucc_logo.png)

UCC is a collective communication operations API and library that is flexible, complete, and feature-rich for current and emerging programming models and runtimes.

- [Design Goals](#design-goals)
- [API](#api)
- [Building](#compiling-and-installing)
- [Community](#community)
- [Contributing](#contributing)
- [License](#license)

## Design Goals
* Highly scalable and performant collectives for HPC, AI/ML and I/O workloads
* Nonblocking collective operations that cover a variety of programming models
* Flexible resource allocation model
* Support for relaxed ordering model
* Flexible synchronous model 
* Repetitive collective operations (init once and invoke multiple times)
* Hardware collectives are a first-class citizen

## Contributing
Thanks for your interest in contributing to UCC, please see our technical and
legal guidelines in the [contributing](CONTRIBUTING.md) file.

## License
UCC is BSD-style licensed, as found in the [LICENSE](LICENSE) file.

## Required packages

* [UCX](https://github.com/openucx/ucx)
   * UCC uses utilities provided by UCX's UCS component
   
* Doxygen
   * UCC uses Doxygen for generating API documentation

## Compiling and Installing

### Developer's Build 
```sh
$./autogen.sh

$./configure --prefix=<ucc-install-path> --with-ucx==<ucx-install-path>

$make 
```

### Build Documentation 
```sh
$./autogen.sh

$./configure --with-docs

$make docs
```
