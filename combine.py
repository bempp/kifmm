import os
import pandas as pd

# Define the directory containing the CSV files

arch="m1"

for geometry in ["uniform", "spheroid", "sphere"]:
    for precision in ["f32", "f64"]:
        for m2l in ["blas", "fft"]:
            input_dir = f"./{arch}/{precision}/{geometry}/{m2l}"

            output_file = f"grid_search_laplace_{precision}_{geometry}_{m2l}_{arch}.csv"

            # Initialize an empty DataFrame to store combined data
            combined_data = pd.DataFrame()

            # Iterate through all files in the directory
            for filename in os.listdir(input_dir):
                if filename.endswith(".csv"):
                    print(filename)
                    file_path = os.path.join(input_dir, filename)
                    data = pd.read_csv(file_path)  # Read the CSV file
                    combined_data = pd.concat([combined_data, data], ignore_index=True)  # Append to the combined DataFrame

            # Save the combined DataFrame to the specified output file
            combined_data.to_csv(output_file, index=False)
            print(f"All CSV files combined and saved to {output_file}")
