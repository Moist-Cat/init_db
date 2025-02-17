#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

# Get input and output file names from arguments
input_file=$1
output_file=$2

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' does not exist."
    exit 1
fi

# Shorten the dataset to 10k rows (excluding the header if it's a CSV/TSV)
head -n 10000 "$input_file" > "$output_file"

# Inform the user that the operation is complete
echo "Dataset shortened to 10,000 rows and saved to '$output_file'."
