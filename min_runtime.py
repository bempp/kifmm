import sys
import pandas as pd

def find_best_configurations_by_l2(file_path, error_thresholds):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Define the parameter columns
    param_cols = ["depth", "surface_diff", "eps", "expansion_order"]
    
    # Check each threshold
    for threshold in error_thresholds:
        # Filter rows meeting relative error criteria
        filtered = df[df["l2_error"] <= threshold]
        
        if filtered.empty:
            print(f"No configuration found with l2_error <= {threshold:.1e}")
        else:
            # Select the one with minimum runtime
            best_row = filtered.loc[filtered["runtime"].idxmin()]
            print(f"\nBest configuration for l2_error <= {threshold:.1e}:")
            print(best_row[param_cols + ["runtime", "setup_time", "l2_error"]])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 best_config_by_l2_error.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    # Define your error thresholds
    error_thresholds = [1e-3, 1e-4, 1e-5]
    find_best_configurations_by_l2(input_file, error_thresholds)

