import os
import json
import matplotlib.pyplot as plt

results_path = "logs/results"

fedavg = []
fedprox = []
alphas = []

for file in os.listdir(results_path):

    with open(os.path.join(results_path, file)) as f:
        data = json.load(f)

    if data["mu"] == 0.0:
        fedavg.append((data["alpha"], data["best_accuracy"]))
    else:
        fedprox.append((data["alpha"], data["best_accuracy"]))

# Sort by alpha
fedavg.sort()
fedprox.sort()

alphas_avg = [x[0] for x in fedavg]
acc_avg = [x[1] for x in fedavg]

alphas_prox = [x[0] for x in fedprox]
acc_prox = [x[1] for x in fedprox]

plt.figure()
plt.plot(alphas_avg, acc_avg, marker='o', label="FedAvg")
plt.plot(alphas_prox, acc_prox, marker='o', label="FedProx")

plt.xlabel("Alpha (Dirichlet)")
plt.ylabel("Best Validation Accuracy (%)")
plt.title("Accuracy vs Heterogeneity")
plt.legend()
plt.grid(True)

plt.savefig("logs/plots/alpha_comparison.png")
plt.show()
