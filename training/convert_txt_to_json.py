import os
import json
import re

EXPERIMENTS_DIR = "experiments"
OUTPUT_DIR = "logs/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_accuracies(filepath):
    accuracies = []
    with open(filepath, "r") as f:
        for line in f:
            match = re.search(r"Global Val Acc:\s*([\d\.]+)%", line)
            if match:
                acc = float(match.group(1))
                accuracies.append(acc)
    return accuracies

for method_folder in os.listdir(EXPERIMENTS_DIR):
    method_path = os.path.join(EXPERIMENTS_DIR, method_folder)

    if not os.path.isdir(method_path):
        continue

    # Determine method name
    if "fedprox" in method_folder:
        method = "fedprox"
        mu = 0.01  # adjust if needed
    else:
        method = "fedavg"
        mu = 0.0

    for filename in os.listdir(method_path):

        if not filename.endswith(".txt"):
            continue

        alpha_match = re.search(r"alpha_(\d\.\d)", filename)
        if not alpha_match:
            continue

        alpha = float(alpha_match.group(1))

        txt_path = os.path.join(method_path, filename)
        accuracies = extract_accuracies(txt_path)

        if len(accuracies) == 0:
            print(f"Skipping {filename} — no accuracies found")
            continue

        results = {
            "method": method,
            "alpha": alpha,
            "mu": mu,
            "round_accuracies": accuracies,
            "best_accuracy": max(accuracies),
            "final_accuracy": accuracies[-1]
        }

        json_filename = f"{method}_alpha_{alpha}"
        if method == "fedprox":
            json_filename += f"_mu_{mu}"

        json_filename += ".json"

        json_path = os.path.join(OUTPUT_DIR, json_filename)

        with open(json_path, "w") as jf:
            json.dump(results, jf, indent=4)

        print(f"Converted {filename} → {json_filename}")

print("\nAll conversions completed.")
