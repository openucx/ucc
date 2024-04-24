if [ "$#" -ne 3 ]; then
    echo "Usage: $0 N BEG END"
    exit 1
fi

N=$1
BEG=$2
END=$3

# Function to generate random number between BEG and END
function generate_random_number {
    echo $(( RANDOM % (END - BEG + 1) + BEG ))
}

# Generate NxN matrix
for (( i=0; i<N; i++ )); do
    for (( j=0; j<N; j++ )); do
        if [ $i -eq $j ]; then
            echo -n "0 "
        else
            echo -n "$(generate_random_number) "
        fi
    done
    echo
done

