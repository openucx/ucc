source ./run/env.sh

PREFIX="$PWD/build"

echo "Prefix: $PREFIX"
echo "with-ucx: $HPCX_UCX_DIR"
echo "with mpi: $MPI_HOME"

./autogen.sh; ./configure --prefix=$PREFIX --with-ucx=$HPCX_UCX_DIR --with-mpi=$MPI_HOME; make -j install
