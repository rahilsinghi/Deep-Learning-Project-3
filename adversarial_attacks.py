import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
MEAN_NORMS = np.array([0.485, 0.456, 0.406])
STD_NORMS = np.array([0.229, 0.224, 0.225])

# Image preprocessing
plain_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_NORMS, std=STD_NORMS)
])

def load_model():
    """Load and prepare the ResNet-34 model"""
    model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    model = model.to(device)
    model.eval()
    return model

def load_dataset(dataset_path):
    """Load the test dataset"""
    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=plain_transforms)
    
    # Debug: Print dataset information
    print("\nDataset information:")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Class names: {dataset.classes}")
    print(f"Class to idx mapping: {dataset.class_to_idx}")
    
    return dataset

def load_imagenet_labels():
    """Load ImageNet class labels and create mappings"""
    with open('TestDataSet/labels_list.json', 'r') as f:
        labels = json.load(f)
    
    # Create mappings
    label_map = {}  # Maps ImageNet index to label name
    class_map = {}  # Maps dataset index to ImageNet index
    
    # First, create mapping of ImageNet indices to labels
    for label in labels:
        idx, name = label.split(': ')
        idx = int(idx)
        label_map[idx] = name
    
    # Create direct mapping from dataset indices to ImageNet indices
    # Since our dataset is ordered from 0-99 and labels are from 401-500,
    # we can create a direct mapping
    for i in range(100):
        class_map[i] = i + 401
    
    # Print debug information
    print("\nMapping Information:")
    print(f"Number of ImageNet labels: {len(label_map)}")
    print(f"Number of class mappings: {len(class_map)}")
    print("\nFirst few class mappings:")
    for i, (k, v) in enumerate(sorted(class_map.items())):
        if i < 5:
            print(f"Dataset idx {k} -> ImageNet idx {v} ({label_map[v]})")
    
    return label_map, class_map

def evaluate_model(model, dataloader, label_map, class_map):
    """Evaluate model performance and return top-1 and top-5 accuracy"""
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    # Debug: Print class mapping information
    print("\nClass mapping information:")
    print(f"Number of classes in class_map: {len(class_map)}")
    print("First few mappings:")
    for i, (k, v) in enumerate(sorted(class_map.items())):
        if i < 5:
            print(f"Dataset index {k} -> ImageNet index {v}")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            outputs = model(images)
            
            # Debug: Print label information for first batch
            if batch_idx == 0:
                print("\nFirst batch label information:")
                print(f"Raw labels: {labels.tolist()}")
                print(f"Unique labels in batch: {torch.unique(labels).tolist()}")
            
            # Map the dataset labels to ImageNet class indices
            try:
                imagenet_labels = torch.tensor([class_map[label.item()] for label in labels], device=device)
            except KeyError as e:
                print(f"\nError: Label {e} not found in class_map")
                print(f"Available labels in class_map: {sorted(class_map.keys())}")
                raise
            
            # Print debug information for first batch
            if batch_idx == 0:
                print("\nFirst batch predictions vs true labels:")
                _, predicted = outputs.max(1)
                for i in range(min(5, len(predicted))):
                    pred_idx = predicted[i].item()
                    true_idx = imagenet_labels[i].item()
                    pred_label = label_map.get(pred_idx, f"Unknown class {pred_idx}")
                    true_label = label_map.get(true_idx, f"Unknown class {true_idx}")
                    print(f"Predicted: {pred_idx} ({pred_label}), True: {true_idx} ({true_label})")
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += images.size(0)
            correct_1 += (predicted == imagenet_labels).sum().item()
            
            # Top-5 accuracy
            _, predicted_5 = outputs.topk(5, 1, largest=True, sorted=True)
            imagenet_labels = imagenet_labels.view(imagenet_labels.size(0), -1).expand_as(predicted_5)
            correct_5 += predicted_5.eq(imagenet_labels).sum().item()
    
    top1_accuracy = 100. * correct_1 / total
    top5_accuracy = 100. * correct_5 / total
    
    return top1_accuracy, top5_accuracy

def fgsm_attack(model, images, labels, epsilon):
    """Implement Fast Gradient Sign Method attack"""
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Create adversarial images
    with torch.no_grad():
        perturbed_images = images + epsilon * torch.sign(images.grad)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
    return perturbed_images

def save_adversarial_images(images, output_dir):
    """Save adversarial images"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Denormalize images
    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    images = images * STD_NORMS + MEAN_NORMS
    images = np.clip(images * 255, 0, 255).astype(np.uint8)
    
    # Save images
    for i, img in enumerate(images):
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(output_dir, f'adv_image_{i}.jpg'))

def create_adversarial_dataset(model, dataloader, epsilon, class_map, output_dir="AdversarialTestSet1"):
    """Create adversarial dataset using FGSM attack"""
    print(f"\nCreating adversarial dataset with epsilon = {epsilon}...")
    
    # Store original model mode and set to eval
    original_mode = model.training
    model.eval()
    
    all_perturbed_images = []
    max_l_inf_norm = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
        images, labels = images.to(device), labels.to(device)
        
        # Map labels to ImageNet indices
        imagenet_labels = torch.tensor([class_map[label.item()] for label in labels], device=device)
        
        # Enable gradients temporarily for attack
        images.requires_grad = True
        
        # Forward pass
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, imagenet_labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial images
        with torch.no_grad():
            perturbed_images = images + epsilon * torch.sign(images.grad)
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        # Calculate L∞ norm
        l_inf_norm = torch.max(torch.abs(perturbed_images - images)).item()
        max_l_inf_norm = max(max_l_inf_norm, l_inf_norm)
        
        # Save perturbed images
        all_perturbed_images.append(perturbed_images.cpu())
        
        # Visualize first batch
        if batch_idx == 0:
            with torch.no_grad():
                adv_outputs = model(perturbed_images)
                _, predicted = adv_outputs.max(1)
                print("\nFirst batch adversarial examples:")
                for i in range(min(3, len(predicted))):
                    print(f"Original class: {imagenet_labels[i].item()}")
                    print(f"Adversarial prediction: {predicted[i].item()}")
    
    # Restore original model mode
    model.train(original_mode)
    
    # Concatenate all perturbed images
    all_perturbed_images = torch.cat(all_perturbed_images, dim=0)
    
    # Save adversarial dataset
    save_adversarial_images(all_perturbed_images, output_dir)
    
    print(f"\nMaximum L∞ norm between original and perturbed images: {max_l_inf_norm:.4f}")
    return all_perturbed_images

def visualize_attack(original_images, perturbed_images, labels, predictions, label_map, num_examples=3):
    """Visualize original and adversarial examples"""
    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 5*num_examples))
    
    for i in range(num_examples):
        # Original image
        orig_img = original_images[i].cpu().numpy().transpose(1, 2, 0)
        orig_img = np.clip(orig_img * STD_NORMS + MEAN_NORMS, 0, 1)
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f'Original\nTrue: {label_map[labels[i]]}\nPred: {label_map[predictions[i]]}')
        axes[i, 0].axis('off')
        
        # Perturbed image
        pert_img = perturbed_images[i].cpu().numpy().transpose(1, 2, 0)
        pert_img = np.clip(pert_img * STD_NORMS + MEAN_NORMS, 0, 1)
        axes[i, 1].imshow(pert_img)
        axes[i, 1].set_title(f'Perturbed\nTrue: {label_map[labels[i]]}\nPred: {label_map[predictions[i]]}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def project_perturbation(perturbation, epsilon):
    """Project perturbation onto L-infinity ball of radius epsilon"""
    return torch.clamp(perturbation, -epsilon, epsilon)

def clip_image_values(images):
    """Clip image values to valid range [0, 1]"""
    return torch.clamp(images, 0, 1)

def pgd_attack(model, images, labels, epsilon, alpha, num_steps):
    """
    Projected Gradient Descent (PGD) attack
    Args:
        model: target model
        images: input images
        labels: true labels
        epsilon: maximum perturbation
        alpha: step size
        num_steps: number of steps
    """
    images = images.clone().detach()
    labels = labels.clone().detach()
    
    # Random initialization within epsilon ball
    delta = torch.zeros_like(images)
    delta = delta + torch.empty_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(delta, -epsilon, epsilon)
    delta.requires_grad = True
    
    for _ in range(num_steps):
        # Forward pass
        outputs = model(images + delta)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update perturbation
        grad = delta.grad.detach()
        delta.data = delta + alpha * torch.sign(grad)
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.data = torch.clamp(images + delta.data, 0, 1) - images
        
        # Zero gradients for next step
        delta.grad.zero_()
    
    return clip_image_values(images + delta.detach())

def ifgsm_attack(model, images, labels, epsilon, alpha, num_steps, momentum=0.9):
    """
    Iterative FGSM attack with momentum
    Args:
        model: target model
        images: input images
        labels: true labels
        epsilon: maximum perturbation
        alpha: step size
        num_steps: number of steps
        momentum: momentum factor
    """
    images = images.clone().detach()
    labels = labels.clone().detach()
    
    # Initialize momentum accumulator
    grad_momentum = torch.zeros_like(images)
    
    # Initialize perturbation
    perturbed_images = images.clone().detach()
    
    for _ in range(num_steps):
        perturbed_images.requires_grad = True
        
        # Forward pass
        outputs = model(perturbed_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update momentum term
        grad = perturbed_images.grad.data
        grad_norm = torch.norm(grad, p=1)
        if grad_norm != 0:
            grad = grad / grad_norm  # L1 normalization
        
        grad_momentum = momentum * grad_momentum + grad
        
        # Update image
        perturbed_images = perturbed_images.detach() + alpha * torch.sign(grad_momentum)
        
        # Project perturbation
        delta = perturbed_images - images
        delta = project_perturbation(delta, epsilon)
        perturbed_images = clip_image_values(images + delta)
    
    return perturbed_images

def create_adversarial_dataset_improved(model, dataloader, attack_params, class_map, output_dir="AdversarialTestSet2"):
    """Create adversarial dataset using improved attacks"""
    print(f"\nCreating adversarial dataset with {attack_params['attack_type']}...")
    print(f"Parameters: {attack_params}")
    
    # Store original model mode and set to eval
    original_mode = model.training
    model.eval()
    
    all_perturbed_images = []
    max_l_inf_norm = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
        images, labels = images.to(device), labels.to(device)
        
        # Map labels to ImageNet indices
        imagenet_labels = torch.tensor([class_map[label.item()] for label in labels], device=device)
        
        # Apply selected attack
        if attack_params['attack_type'] == 'pgd':
            perturbed_images = pgd_attack(
                model, images, imagenet_labels,
                epsilon=attack_params['epsilon'],
                alpha=attack_params['alpha'],
                num_steps=attack_params['num_steps']
            )
        elif attack_params['attack_type'] == 'ifgsm':
            perturbed_images = ifgsm_attack(
                model, images, imagenet_labels,
                epsilon=attack_params['epsilon'],
                alpha=attack_params['alpha'],
                num_steps=attack_params['num_steps'],
                momentum=attack_params['momentum']
            )
        
        # Calculate L∞ norm
        l_inf_norm = torch.max(torch.abs(perturbed_images - images)).item()
        max_l_inf_norm = max(max_l_inf_norm, l_inf_norm)
        
        # Save perturbed images
        all_perturbed_images.append(perturbed_images.cpu())
        
        # Visualize first batch
        if batch_idx == 0:
            with torch.no_grad():
                outputs = model(images)
                adv_outputs = model(perturbed_images)
                _, predicted = outputs.max(1)
                _, adv_predicted = adv_outputs.max(1)
                print("\nFirst batch examples:")
                for i in range(min(3, len(predicted))):
                    print(f"Original prediction: {predicted[i].item()}")
                    print(f"Adversarial prediction: {adv_predicted[i].item()}")
    
    # Restore original model mode
    model.train(original_mode)
    
    # Concatenate all perturbed images
    all_perturbed_images = torch.cat(all_perturbed_images, dim=0)
    
    # Save adversarial dataset
    save_adversarial_images(all_perturbed_images, output_dir)
    
    print(f"\nMaximum L∞ norm between original and perturbed images: {max_l_inf_norm:.4f}")
    return all_perturbed_images

def main():
    # Load model
    model = load_model()
    
    # Load dataset and labels
    dataset_path = "./TestDataSet"
    dataset = load_dataset(dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load ImageNet labels and class mappings
    label_map, class_map = load_imagenet_labels()
    
    # Create reverse mapping for evaluation
    reverse_class_map = {v: k for k, v in class_map.items()}
    
    # Task 1: Evaluate baseline performance
    print("\nTask 1: Evaluating baseline performance...")
    top1_acc, top5_acc = evaluate_model(model, dataloader, label_map, class_map)
    print(f"Baseline Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Baseline Top-5 Accuracy: {top5_acc:.2f}%")
    
    # Task 2: Create and evaluate adversarial dataset using FGSM
    print("\nTask 2: Creating adversarial dataset using FGSM...")
    epsilon = 0.02
    adversarial_images = create_adversarial_dataset(model, dataloader, epsilon, class_map)
    
    # Create a new dataloader for adversarial images
    adversarial_dataset = torch.utils.data.TensorDataset(
        adversarial_images.to(device),
        torch.tensor([reverse_class_map[class_map[label]] for label in dataset.targets], device=device)
    )
    adversarial_dataloader = torch.utils.data.DataLoader(
        adversarial_dataset, batch_size=32, shuffle=False
    )
    
    # Evaluate model on FGSM adversarial dataset
    print("\nEvaluating model on FGSM adversarial dataset...")
    adv_top1_acc, adv_top5_acc = evaluate_model(model, adversarial_dataloader, label_map, class_map)
    print(f"FGSM Adversarial Top-1 Accuracy: {adv_top1_acc:.2f}%")
    print(f"FGSM Adversarial Top-5 Accuracy: {adv_top5_acc:.2f}%")
    
    # Task 3: Create and evaluate adversarial datasets using improved attacks
    print("\nTask 3: Testing improved attacks...")
    
    # PGD Attack
    pgd_params = {
        'attack_type': 'pgd',
        'epsilon': 0.02,
        'alpha': 0.002,  # Step size (1/10 of epsilon)
        'num_steps': 10
    }
    
    pgd_images = create_adversarial_dataset_improved(model, dataloader, pgd_params, class_map, "AdversarialTestSet_PGD")
    pgd_dataset = torch.utils.data.TensorDataset(
        pgd_images.to(device),
        torch.tensor([reverse_class_map[class_map[label]] for label in dataset.targets], device=device)
    )
    pgd_dataloader = torch.utils.data.DataLoader(pgd_dataset, batch_size=32, shuffle=False)
    
    print("\nEvaluating model on PGD adversarial dataset...")
    pgd_top1_acc, pgd_top5_acc = evaluate_model(model, pgd_dataloader, label_map, class_map)
    print(f"PGD Adversarial Top-1 Accuracy: {pgd_top1_acc:.2f}%")
    print(f"PGD Adversarial Top-5 Accuracy: {pgd_top5_acc:.2f}%")
    
    # I-FGSM Attack with momentum
    ifgsm_params = {
        'attack_type': 'ifgsm',
        'epsilon': 0.02,
        'alpha': 0.002,
        'num_steps': 10,
        'momentum': 0.9
    }
    
    ifgsm_images = create_adversarial_dataset_improved(model, dataloader, ifgsm_params, class_map, "AdversarialTestSet_IFGSM")
    ifgsm_dataset = torch.utils.data.TensorDataset(
        ifgsm_images.to(device),
        torch.tensor([reverse_class_map[class_map[label]] for label in dataset.targets], device=device)
    )
    ifgsm_dataloader = torch.utils.data.DataLoader(ifgsm_dataset, batch_size=32, shuffle=False)
    
    print("\nEvaluating model on I-FGSM adversarial dataset...")
    ifgsm_top1_acc, ifgsm_top5_acc = evaluate_model(model, ifgsm_dataloader, label_map, class_map)
    print(f"I-FGSM Adversarial Top-1 Accuracy: {ifgsm_top1_acc:.2f}%")
    print(f"I-FGSM Adversarial Top-5 Accuracy: {ifgsm_top5_acc:.2f}%")

if __name__ == "__main__":
    main() 