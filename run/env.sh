export UCC_PT_COLL_ALLTOALLV_TRANSFER_MATRIX_FILE="$PWD/run/transfer_matrix"
export \
	UCX_TLS=rc,cuda_copy \
	UCX_RNDV_SCHEME=get_zcopy \
	MELLANOX_VISIBLE_DEVICES=0,3,4,5,6,9,10,11 \
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
	UCX_IB_GID_INDEX=3  \
	UCX_RNDV_THRESH=0 \
	OMPI_MCA_btl=tcp,self \
	OMPI_MCA_btl_tcp_if_include=eno8303 

module purge

#module use /hpc/local/etc/modulefiles/dev
#module load nccl_2.18.5-1_cuda12.2.2_x86_64 
#module load cuda12.2.2

module use /hpc/local/etc/modulefiles
module load hpcx-gcc

module list

