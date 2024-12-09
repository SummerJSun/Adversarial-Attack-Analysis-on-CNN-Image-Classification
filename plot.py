import matplotlib.pyplot as plt

epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
accuracy_drops = [1.01, 0.95, 3.69, 28.96, 57.85, 88.77]

plt.figure(figsize=(8, 6))
plt.plot(epsilons, accuracy_drops, marker='o', linestyle='-', linewidth=2)
plt.title("Accuracy Drop vs. Epsilon For ResNet 18", fontsize=16)
plt.xlabel("Epsilon", fontsize=14)
plt.ylabel("Accuracy Drop (%)", fontsize=14)
plt.xscale('log')
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

plt.savefig("accuracy_drop_vs_epsilon_res.jpg", dpi=300)
plt.show()
