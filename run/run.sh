source ./run/env.sh

srun \
	--mpi=pmix \
	--ntasks-per-node=8 \
	--gpus-per-node=8 \
	-N 4 \
	-p ISR1-ALL \
	#-o "ucc_alltoallv_$(date +%s).log" \
	./build/bin/ucc_perftest -c alltoallv 

