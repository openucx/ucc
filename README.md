# Unified Collective Communication (UCC)

<img src="docs/images/ucc_logo.png" width="75%" height="75%">

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

### UCC Component Architecture
![](docs/images/ucc_components.png)

## Contributing
Thanks for your interest in contributing to UCC, please see our technical and
legal guidelines in the [contributing](CONTRIBUTING.md) file.

All contributors have to comply with ["Membership Voluntary
Consensus Standard"](https://ucfconsortium.org/policy/)  and ["Export Compliant
Contribution Submissions"](https://ucfconsortium.org/policy/) policies.

## License
UCC is BSD-style licensed, as found in the [LICENSE](LICENSE) file.

## Required packages

* [UCX](https://github.com/openucx/ucx)
   * UCC uses utilities provided by UCX's UCS component

* [CUDA](https://developer.nvidia.com/cuda-toolkit) (optional)
   * UCC supports CUDA collectives. To compile with CUDA support, install
     [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 11.0 or above.

* [HIP](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html) (optional)
   * UCC supports AMD GPUs using HIP. Instructions for installing ROCM/HIP can be found at
     [AMD ROCM](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation_new.html).

* Doxygen
   * UCC uses Doxygen for generating API documentation

## Compiling and Installing

### Developer's Build
```sh
$ ./autogen.sh
$ ./configure --prefix=<ucc-install-path> --with-ucx=<ucx-install-path>
$ make
```

### Build Documentation
```sh
$ ./autogen.sh
$ ./configure --prefix=<ucc-install-path> --with-docs-only
$ make docs
```

### Open MPI and UCC collectives

#### Compile UCX 
```sh
$ git clone https://github.com/openucx/ucx
$ cd ucx
$ ./autogen.sh; ./configure --prefix=<ucx-install-path>; make -j install
```
#### Compile UCC

```sh
$ git clone https://github.com/openucx/ucc
$ cd ucc
$ ./autogen.sh; ./configure --prefix=<ucc-install-path> --with-ucx=<ucx-install-path>; make -j install
```

#### Compile Open MPI 

```sh
$ git clone https://github.com/open-mpi/ompi
$ cd ompi
$ ./autogen.pl; ./configure --prefix=<ompi-install-path> --with-ucx=<ucx-install-path> --with-ucc=<ucc-install-path>; make -j install
```

#### Run MPI programs

```sh
$ mpirun -np 2 --mca coll_ucc_enable 1 --mca coll_ucc_priority 100 ./my_mpi_app
```

#### Run OpenSHMEM programs

```sh
$ mpirun -np 2 --mca scoll_ucc_enable 1 --mca scoll_ucc_priority 100 ./my_openshmem_app
```


### SUPPORTED Transports
* UCX/UCP
  - InfiniBand, ROCE, Cray Gemini and Aries, Shared Memory
* SHARP
* CUDA
* NCCL
* RCCL
