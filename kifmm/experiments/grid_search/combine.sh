#!/bin/bash

# Directory containing the CSV files
csv_dir="."
# Output file
output_file="combined.csv"

# Check if the output file already exists and remove it
if [ -f "$output_file" ]; then
    rm "$output_file"
fi

# Initialize a variable to keep track of whether the header has been added
header_added=false

# Loop through all CSV files in the directory
for csv_file in "$csv_dir"/grid_search_laplace_fft_f64_m1_**.csv; do
    if [ "$header_added" = false ]; then
        # Add header from the first file
        head -n 1 "$csv_file" > "$output_file"
        header_added=true
    fi
    # Skip the header row and append the rest to the output file
    tail -n +2 "$csv_file" >> "$output_file"
done

echo "CSV files have been combined into $output_file"
