import matplotlib.pyplot as plt

def plot_one():

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
    
def plot_accuracy_drop():
    eps_resnet = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    acc_drop_resnet = [1.01, 0.95, 3.69, 28.96, 57.85, 88.77]

    eps_vit = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    acc_drop_vit = [0.39, 0.36, 1.15, 5.12, 10.97, 36.18]

    eps_clip = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    acc_drop_clip = [0.64, 0.61, 2.13, 16.3, 37.07, 78.21]

    plt.figure(figsize=(10, 6))
    plt.plot(eps_resnet, acc_drop_resnet, marker='o', label="ResNet18")
    plt.plot(eps_vit, acc_drop_vit, marker='s', label="Vision Transformer")
    plt.plot(eps_clip, acc_drop_clip, marker='^', label="CLIP")
    
    plt.xscale('log')
    plt.xlabel('Epsilon (log scale)')
    plt.ylabel('Accuracy Drop (%)')
    plt.title('Adversarial Test Set Accuracy Drop vs Epsilon')
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig('accuracy_drop_vs_epsilon.jpg')
    plt.show()

plot_accuracy_drop()

