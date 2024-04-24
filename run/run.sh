source ./run/env.sh


UCC_LOG_LEVEL=debug; UCX_LOG_LEVEL=debug; UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRIX_FILE="$PWD/run/transfer_matrix_8"; QP=4;PWC=16;N=85;PW=$PWC;UCX_IB_NUM_PATHS=$QP UCX_TLS=rc,cuda_copy UCX_RNDV_SCHEME=get_zcopy MELLANOX_VISIBLE_DEVICES=0,3,4,5,6,9,10,11 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 UCX_IB_GID_INDEX=3  UCC_CLS=basic UCX_RNDV_THRESH=0 UCC_TLS=ucp OMPI_MCA_btl=tcp,self OMPI_MCA_btl_tcp_if_include=eno8303 srun -p ISR1-ALL -N 1 --ntasks-per-node=8 --gpus-per-node=8 --mpi=pmix build/bin/ucc_perftest -c alltoallv 


