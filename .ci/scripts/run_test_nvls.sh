OMPI_MCA_coll=^hcoll \
OMPI_MCA_coll_ucc_enable=0 \
UCC_TLS=cuda,ucp UCC_LOG_LEVEL=info UCC_TL_CUDA_NVLS_SM_COUNT=20 UCC_TL_CUDA_TUNE=allreduce:cuda:@0 \
/opt/nvidia/bin/ucc/build/bin/ucc_perftest -c allreduce -F -m cuda -b 1k -e 32M -d bfloat16 -o sum
