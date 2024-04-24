source ./run/env.sh

PREFIX="$PWD/build"
DEBUG=yes

echo "Prefix: $PREFIX"
echo "with-ucx: $HPCX_UCX_DIR"
echo "with mpi: $MPI_HOME"

./autogen.sh; ./configure --enable-debug=$DEBUG --prefix=$PREFIX --with-ucx=$HPCX_UCX_DIR --with-mpi=$MPI_HOME; make -j install
