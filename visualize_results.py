import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import json
from tqdm import tqdm
from torchvision.datasets import ImageFolder

# Constants
MEAN_NORMS = np.array([0.485, 0.456, 0.406])
STD_NORMS = np.array([0.229, 0.224, 0.225])

def load_imagenet_labels():
    """Load ImageNet class labels"""
    with open('TestDataSet/labels_list.json', 'r') as f:
        labels = json.load(f)
    label_map = {}
    for label in labels:
        idx, name = label.split(': ')
        label_map[int(idx)] = name
    return label_map

def plot_accuracy_comparison():
    """Plot accuracy comparison for all attacks"""
    attacks = ['Clean', 'FGSM', 'PGD', 'I-FGSM', 'Patch-PGD']
    top1_acc = [76.00, 26.40, 2.00, 1.60, 33.40]
    top5_acc = [94.20, 50.60, 13.60, 10.20, 55.60]
    
    x = np.arange(len(attacks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, top1_acc, width, label='Top-1 Accuracy')
    rects2 = ax.bar(x + width/2, top5_acc, width, label='Top-5 Accuracy')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Under Different Attacks')
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.legend()
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison_all_attacks.png')
    plt.close()

def plot_attack_effectiveness():
    """Plot attack effectiveness (accuracy reduction)"""
    attacks = ['FGSM', 'PGD', 'I-FGSM', 'Patch-PGD']
    top1_reduction = [76.00 - 26.40, 76.00 - 2.00, 76.00 - 1.60, 76.00 - 33.40]
    top5_reduction = [94.20 - 50.60, 94.20 - 13.60, 94.20 - 10.20, 94.20 - 55.60]
    
    x = np.arange(len(attacks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, top1_reduction, width, label='Top-1 Reduction')
    rects2 = ax.bar(x + width/2, top5_reduction, width, label='Top-5 Reduction')
    
    ax.set_ylabel('Accuracy Reduction (%)')
    ax.set_title('Attack Effectiveness (Accuracy Reduction)')
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('attack_effectiveness.png')
    plt.close()

def plot_patch_attack_examples():
    """Plot examples of patch attacks"""
    label_map = load_imagenet_labels()
    patch_dir = "AdversarialTestSet3"
    if not os.path.exists(patch_dir):
        print(f"Directory {patch_dir} not found. Skipping patch visualization.")
        return

    # Load dataset to get original image paths
    dataset = ImageFolder("TestDataSet", transform=None)
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
    for i in range(5):
        orig_img = Image.open(dataset.samples[i][0])
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        patch_img = Image.open(f"{patch_dir}/adv_image_{i}.jpg")
        axes[i, 1].imshow(patch_img)
        axes[i, 1].set_title(f'Patch-Attacked Image {i+1}')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig('patch_attack_examples.png')
    plt.close()

def plot_attack_parameters():
    """Plot attack parameters comparison"""
    attacks = ['FGSM', 'PGD', 'I-FGSM', 'Patch-PGD']
    epsilons = [0.02, 0.02, 0.02, 0.3]
    steps = [1, 10, 10, 20]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Epsilon values
    ax1.bar(attacks, epsilons)
    ax1.set_ylabel('Epsilon (ε)')
    ax1.set_title('Perturbation Budget (ε)')
    for i, v in enumerate(epsilons):
        ax1.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Number of steps
    ax2.bar(attacks, steps)
    ax2.set_ylabel('Number of Steps')
    ax2.set_title('Attack Iterations')
    for i, v in enumerate(steps):
        ax2.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('attack_parameters.png')
    plt.close()

def main():
    # Generate all visualizations
    print("Generating visualizations...")
    plot_accuracy_comparison()
    plot_attack_effectiveness()
    plot_patch_attack_examples()
    plot_attack_parameters()
    print("Visualizations saved!")

if __name__ == "__main__":
    main() 