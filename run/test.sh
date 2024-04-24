#!/bin/bash

# Number of times to call the scripts
ITERATIONS=1

# Loop for ITERATIONS times
for (( i=1; i<=ITERATIONS; i++ )); do
    echo "Iteration $i:"
    
    # Call generate_matrix.sh
    echo "Generating matrix..."
    bash run/generate_matrix.sh 8 10000 10000000 | tee run/transfer_matrix_8
    
    # Call test.sh
    echo "Testing matrix..."
    bash run/run.sh
    
    echo "---------------------------------"
done
