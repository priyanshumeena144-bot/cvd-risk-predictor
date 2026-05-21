import subprocess
import os

# Files jin par metrics nikalni hai
target_files = ["main.py", "app.py"] 

# PyMetrics ko seedhe run karne ka command
command = ["pymetrics"] + target_files

print(f"Attempting to run command: {' '.join(command)}")

try:
    # Command ko execute karein aur output ko capture karein
    result = subprocess.run(
        command, 
        capture_output=True, 
        text=True, 
        check=True,
        shell=True  # Windows mein shell=True path issues ko bypass karta hai
    )
    print("\n--- PYMETRICS OUTPUT ---")
    print(result.stdout)
    print("------------------------\n")

except subprocess.CalledProcessError as e:
    print(f"\nERROR: PyMetrics could not be executed.")
    print(f"Check if 'pymetrics' is installed correctly (pip install pymetrics).")
    print(f"Detailed Error: {e.stderr}")
    
except FileNotFoundError:
    print(f"\nFATAL ERROR: 'pymetrics' command was not found by the OS.")

# Final Viva Prep: OO Metrics ke principles samjhein
print("\nOO Metrics (CBO, LCOM) ke principles samajhkar Viva dein.")