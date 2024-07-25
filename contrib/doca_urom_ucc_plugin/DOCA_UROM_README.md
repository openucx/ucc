# How to enable and run collectives on NVIDIA DPUs
The CL_DOCA_UROM plugin enables collective offloads via DOCA and CL_DOCA_UROM feature. The plugin runs on the DPU and is initialized via the command channel from CL_DOCA_UROM. It leverages UCC collectives optimized for the DPU and facilitates efficient computation and communication overlap. While the plugin supports all UCC collectives via a copy-in/out method, the optimized algorithms are Allreduce and Alltoall/v (OpenSHMEM).

## Components and Dependencies

### Components
1. **Host Node:**
    - Runs the main UCC application and initiates communication with the DPU.
    - Executes scripts to run the UCC collective operations with DOCA integration.
2. **DPU:**
    - Runs the DOCA services.
    - Leverages UCC for data handling.
    - Communicates with the host node using the DOCA interface.

### Interaction
- The host node communicates with the DPU via the DOCA interface.
- Data and commands are sent from the host to the DPU, processed by UCC, and results are returned.

### Dependencies
- **Programming Languages:** C/C++, Shell scripting
- **Libraries:** UCC, UCX (Unified Communication X)
- **Tools:** DOCA SDK, OpenMPI
- **Platforms:** Host node (x86 or ARM), DPU (NVIDIA BlueField)

### Interface Design
- **APIs:** UCC collective operations with DOCA_UROM extensions.

## Build and Deployment

For DOCA instructions please see: [DOCA - Get Started | NVIDIA Developer](https://docs.nvidia.com/doca/sdk/nvidia+doca+overview/index.html)

### Host Node
1. **Build Instructions:**
    ```
    Checkout UCC
    ./autogen.sh
    ./configure --prefix=<path> --with-ucx=<path to ucx>  --with-doca_urom=<path to doca>
    make -j install
    ```

### DPU
1. **Build Instructions:**
    - Checkout the same UCC branch as on the host.
    - Use the same configure line as the host.

### Generate the following files in $TOP
1. **MPI HOST hostfile:**
    ```plaintext
    hostA slots=1
    hostB slots=1
    ```
2. **MPI DPU hostfile:**
    ```plaintext
    dpuA slots=1
    dpuB slots=1
    ```
3. **DPU servicefile:**
    ```plaintext
    hostA dpu dpuA 0000:04:00.0 mlx5_0:1
    hostB dpu dpuB 0000:04:00.0 mlx5_0:1
    ```

## Run Instructions

### 1. Run on DPU:
- **Script:** `run_doca_uromd.sh`
    ```bash
    #!/bin/bash

    TOP=<working dir>

    OMPI_DIR=<ompi dir>
    DPU_HOST_FILE=<dpu hostfile>

    NUMBER_OF_NODES=$(cat $DPU_FILE | grep -v '#' | wc -l)

    # Set plugin path
    export UROM_PLUGIN_PATH=<path to ucc doca_plugins>
    export PATH=$OMPI_DIR/bin:$PATH
    export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH

    export UCC_DIR=<ucc>
    export UCX_DIR=<ucx>
    export LD_LIBRARY_PATH=$UCC_DIR/lib:$UCX_DIR/lib:$LD_LIBRARY_PATH

    options="-x UCX_TLS=rc_x,tcp $options"

    mpirun --tag-output -np $NUMBER_OF_NODES -hostfile $DPU_HOST_FILE -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH $options -x UROM_PLUGIN_PATH=$UROM_PLUGIN_PATH $TOP/doca/build-dpu/services/urom/doca_urom_daemon -l 10 --sdk-log-level 10
    ```

### 2. Run on Host:
- **Script:** `run_doca_urom_cl.sh`
    ```bash
    #!/bin/bash

    PPN=$1

    if [ -z "$PPN" ]; then
        echo PPN not set, assuming PPN=1
        PPN=1
    fi

    TOP=<path to build dir>

    NUMBER_OF_NODES=$(cat $TOP/hostfile | grep -v '#' | wc -l)

    OMPI_DIR=<ompi dir>
    UCX_DIR=<ucx dir>
    export LD_LIBRARY_PATH=$OMPI_DIR/lib:$UCX_DIR/lib:$UCX/lib/ucx:$TOP/<path to doca>/doca/lib64:$TOP/doca_urom_ucc/install/host/lib64:$LD_LIBRARY_PATH

    OMB_DIR=<path to osu benchmarks>

    # DPU options
    options="$options -x UCX_NET_DEVICES=mlx5_0:1,mn0"
    options="$options -x DOCA_UROM_SERVICE_FILE=$TOP/servicefile"
    options="$options -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

    # `UCC_CL_DOCA_UROM_PLUGIN_ENVS` takes a comma separated list of envs
    options="$options -x UCC_CL_DOCA_UROM_PLUGIN_ENVS=LD_LIBRARY_PATH=$TOP/arm/build-arm/ompi/lib:$TOP/arm/build-arm/ucx/lib:$TOP/arm/build-arm/ucc/lib:$TOP/doca_urom_ucc/install/dpu/lib64,UCX_LOG_LEVEL=ERROR"

    options="$options --mca coll_ucc_enable 1 --mca coll_ucc_priority 100"
    options="$options -x UCX_TLS=rc_x,tcp"

    validation="--validation" #Optional argument to run data validation in osu benchmark
    msg_low=1024
    msg_high=$((1024*1024))

    for coll in allreduce
    do
        $OMPI/bin/mpirun -np $((NODES * PPN)) --map-by node -hostfile $TOP/hostfile $options --mca coll_ucc_cls doca_urom,basic --tag-output $OMB/osu_$coll $validation "-m $msg_low:$msg_high" -i 10 -x 5
    done
    ```

## Key Environment Variables

### Host Side
- `UCC_CL_DOCA_UROM_PLUGIN_ENVS`: Ensures the plugin has the correct `LD_LIBRARY_PATH`.
- `DOCA_UROM_SERVICE_FILE`: Maps between host and DPUs.

### DPU Side
- `UROM_PLUGIN_PATH`: Path to the directory containing only `.so` files that are plugins. A plugin must have the symbols `urom_plugin_get_version` and `urom_plugin_get_iface`.

## Conclusion
The CL_DOCA_UROM feature and the corresponding plugin integrate the UCC collective library with DOCA running on DPUs, leveraging optimized algorithms for the DPU. This README outlines the components, interaction, build, deployment, and execution instructions necessary to implement, run, and extend this feature.
