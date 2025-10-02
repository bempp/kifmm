import re

# Read file
with open("weak_fft_11004445.csv", "r") as f:
    data = f.read()

# Replace "digit + space + digit" with "digit, digit"
fixed_data = re.sub(r"(\d)\s+(\d)", r"\1,\2", data)

# Write corrected CSV
with open("weak_fft_11004445.csv", "w") as f:
    f.write(fixed_data)