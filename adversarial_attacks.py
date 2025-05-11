"""
Adversarial Attacks Implementation

This module implements various adversarial attacks on deep learning models:
- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)
- Iterative Fast Gradient Sign Method (I-FGSM)
- Patch-based PGD Attack

Each attack class provides methods to generate adversarial examples and evaluate their effectiveness.
"""

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

def load_model() -> nn.Module:
    """
    Load and prepare the ResNet-34 model.
    
    Returns:
        nn.Module: Pre-trained ResNet-34 model loaded with ImageNet weights.
    """
    model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    model = model.to(device)
    model.eval()
    return model

def load_dataset(dataset_path: str) -> torchvision.datasets.ImageFolder:
    """
    Load the test dataset from the specified path.
    
    Args:
        dataset_path (str): Path to the dataset directory.
        
    Returns:
        torchvision.datasets.ImageFolder: Loaded dataset.
    """
    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=plain_transforms)
    
    print("\nDataset information:")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Class names: {dataset.classes}")
    print(f"Class to idx mapping: {dataset.class_to_idx}")
    
    return dataset

def load_imagenet_labels() -> tuple[dict, dict]:
    """
    Load ImageNet class labels and create mappings.
    
    Returns:
        tuple[dict, dict]: A tuple containing:
            - label_map: Maps ImageNet index to label name
            - class_map: Maps dataset index to ImageNet index
    """
    with open('TestDataSet/labels_list.json', 'r') as f:
        labels = json.load(f)
    
    label_map = {}
    class_map = {}
    
    for label in labels:
        idx, name = label.split(': ')
        idx = int(idx)
        label_map[idx] = name
    
    for i in range(100):
        class_map[i] = i + 401
    
    print("\nMapping Information:")
    print(f"Number of ImageNet labels: {len(label_map)}")
    print(f"Number of class mappings: {len(class_map)}")
    
    return label_map, class_map

def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    label_map: dict,
    class_map: dict
) -> tuple[float, float]:
    """
    Evaluate model performance and return top-1 and top-5 accuracy.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader containing evaluation data
        label_map: Mapping of indices to class names
        class_map: Mapping of dataset indices to ImageNet indices
        
    Returns:
        tuple[float, float]: Top-1 and Top-5 accuracy percentages
    """
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            
            imagenet_labels = torch.tensor([class_map[label.item()] for label in labels], device=device)
            
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

class FGSMAttack:
    """
    Fast Gradient Sign Method (FGSM) Attack Implementation.
    
    FGSM is a single-step attack that perturbs the input image in the direction
    of the gradient of the loss function with respect to the input.
    
    Attributes:
        model (nn.Module): The target model to attack
        epsilon (float): Maximum perturbation size
        device (torch.device): Device to run the attack on (CPU/GPU)
    """
    
    def __init__(self, model: nn.Module, epsilon: float = 0.03):
        """
        Initialize FGSM attack.
        
        Args:
            model: Target model to attack
            epsilon: Maximum perturbation size (default: 0.03)
        """
        self.model = model
        self.epsilon = epsilon
        self.device = next(model.parameters()).device
        
    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial examples using FGSM.
        
        Args:
            images: Input images to perturb
            labels: True labels for the images
            
        Returns:
            Adversarial examples
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        images.requires_grad = True
        outputs = self.model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        loss.backward()
        perturbation = self.epsilon * torch.sign(images.grad.data)
        adversarial_images = images + perturbation
        
        return torch.clamp(adversarial_images, 0, 1)

class PGDAttack:
    """
    Projected Gradient Descent (PGD) Attack Implementation.
    
    PGD is an iterative attack that performs multiple steps of FGSM with
    projection to ensure the perturbation stays within the epsilon ball.
    
    Attributes:
        model (nn.Module): The target model to attack
        epsilon (float): Maximum perturbation size
        alpha (float): Step size for each iteration
        steps (int): Number of iterations
        device (torch.device): Device to run the attack on (CPU/GPU)
    """
    
    def __init__(self, model: nn.Module, epsilon: float = 0.03, 
                 alpha: float = 0.01, steps: int = 10):
        """
        Initialize PGD attack.
        
        Args:
            model: Target model to attack
            epsilon: Maximum perturbation size (default: 0.03)
            alpha: Step size for each iteration (default: 0.01)
            steps: Number of iterations (default: 10)
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.device = next(model.parameters()).device
        
    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial examples using PGD.
        
        Args:
            images: Input images to perturb
            labels: True labels for the images
            
        Returns:
            Adversarial examples
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        adversarial_images = images.clone().detach()
        
        for _ in range(self.steps):
            adversarial_images.requires_grad = True
            outputs = self.model(adversarial_images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            loss.backward()
            perturbation = self.alpha * torch.sign(adversarial_images.grad.data)
            adversarial_images = adversarial_images + perturbation
            
            # Project back to epsilon ball
            delta = adversarial_images - images
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            adversarial_images = torch.clamp(images + delta, 0, 1)
            
        return adversarial_images

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

def adversarial_training_step(model, images, labels, optimizer, epsilon, alpha, num_steps):
    """
    Perform one step of adversarial training using PGD
    Args:
        model: model to train
        images: input images
        labels: true labels
        optimizer: optimizer for model parameters
        epsilon: maximum perturbation
        alpha: step size for PGD
        num_steps: number of PGD steps
    """
    model.train()
    
    # Generate adversarial examples using PGD
    perturbed_images = pgd_attack(model, images, labels, epsilon, alpha, num_steps)
    
    # Forward pass on adversarial examples
    outputs = model(perturbed_images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def collect_metrics(model, dataloader, label_map, class_map, attack_type="clean"):
    """
    Collect detailed metrics for model evaluation
    Args:
        model: model to evaluate
        dataloader: data loader for evaluation
        label_map: mapping of indices to class names
        class_map: mapping of dataset indices to ImageNet indices
        attack_type: type of attack used (for logging)
    Returns:
        Dictionary containing various metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    total = 0
    correct = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating {attack_type}"):
            images, labels = images.to(device), labels.to(device)
            imagenet_labels = torch.tensor([class_map[label.item()] for label in labels], device=device)
            
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(imagenet_labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            
            total += images.size(0)
            correct += (predictions == imagenet_labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_confidence = np.mean(all_confidences)
    
    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'confidences': np.array(all_confidences)
    }

def plot_training_curves(train_losses, save_path='training_curves.png'):
    """
    Plot training curves
    Args:
        train_losses: list of training losses
        save_path: path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Adversarial Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_comparison(metrics_dict, save_path='accuracy_comparison.png'):
    """
    Plot accuracy comparison between different attacks
    Args:
        metrics_dict: dictionary containing metrics for different attacks
        save_path: path to save the plot
    """
    attacks = list(metrics_dict.keys())
    accuracies = [metrics_dict[attack]['accuracy'] for attack in attacks]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(attacks, accuracies)
    plt.xlabel('Attack Type')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Under Different Attacks')
    plt.ylim(0, 100)
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.savefig(save_path)
    plt.close()

def plot_confidence_distribution(metrics_dict, save_path='confidence_distribution.png'):
    """
    Plot confidence distribution for different attacks
    Args:
        metrics_dict: dictionary containing metrics for different attacks
        save_path: path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for attack_type, metrics in metrics_dict.items():
        plt.hist(metrics['confidences'], bins=20, alpha=0.5, label=attack_type)
    
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution Under Different Attacks')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_metrics_to_file(metrics_dict, filename='task4_metrics.txt'):
    """
    Save metrics to a text file
    Args:
        metrics_dict: dictionary containing metrics for different attacks
        filename: name of the file to save metrics
    """
    with open(filename, 'w') as f:
        f.write("Adversarial Training Results\n")
        f.write("==========================\n\n")
        
        for attack_type, metrics in metrics_dict.items():
            f.write(f"{attack_type.upper()} Attack:\n")
            f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
            f.write(f"Average Confidence: {metrics['avg_confidence']:.4f}\n")
            f.write("\n")

def train_adversarial_model(model, train_loader, num_epochs, learning_rate, epsilon, alpha, num_steps):
    """
    Train model using adversarial training
    Args:
        model: model to train
        train_loader: training data loader
        num_epochs: number of training epochs
        learning_rate: learning rate for optimizer
        epsilon: maximum perturbation
        alpha: step size for PGD
        num_steps: number of PGD steps
    Returns:
        Trained model and training losses
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            images, labels = images.to(device), labels.to(device)
            
            # Perform adversarial training step
            loss = adversarial_training_step(model, images, labels, optimizer, epsilon, alpha, num_steps)
            
            total_loss += loss
            train_losses.append(loss)
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss:.4f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    return model, train_losses

def targeted_pgd_patch_attack(model, images, labels, patch_size=32, epsilon=0.3, alpha=0.03, num_steps=20, num_classes=100):
    """
    Patch-based targeted PGD attack: only a random 32x32 patch is perturbed, and the attack is targeted.
    Args:
        model: target model
        images: input images (N, C, H, W)
        labels: true labels (ImageNet indices)
        patch_size: size of the square patch to perturb
        epsilon: maximum perturbation
        alpha: step size
        num_steps: number of PGD steps
        num_classes: number of classes (for random target selection)
    Returns:
        Adversarial images
    """
    images = images.clone().detach()
    N, C, H, W = images.shape
    device = images.device

    # For each image, pick a random patch location and a random target class (not the true label)
    patch_coords = []
    target_labels = []
    for i in range(N):
        x = np.random.randint(0, W - patch_size + 1)
        y = np.random.randint(0, H - patch_size + 1)
        patch_coords.append((x, y))
        # Pick a random target class different from the true label
        true_label = labels[i].item()
        possible_targets = list(set(range(num_classes)) - {true_label})
        target_label = np.random.choice(possible_targets)
        target_labels.append(target_label)
    target_labels = torch.tensor(target_labels, device=device)

    # Initialize delta (perturbation) as zeros
    delta = torch.zeros_like(images)
    delta.requires_grad = True

    for step in range(num_steps):
        adv_images = images + delta
        outputs = model(adv_images)
        # Targeted attack: minimize loss towards the target class
        loss = nn.CrossEntropyLoss()(outputs, target_labels)
        model.zero_grad()
        loss.backward()

        # Only update the patch region for each image
        grad = delta.grad.detach()
        for i in range(N):
            x, y = patch_coords[i]
            # Update only the patch
            delta.data[i, :, y:y+patch_size, x:x+patch_size] -= alpha * torch.sign(grad[i, :, y:y+patch_size, x:x+patch_size])
            # Project patch perturbation to epsilon ball
            delta.data[i, :, y:y+patch_size, x:x+patch_size] = torch.clamp(
                delta.data[i, :, y:y+patch_size, x:x+patch_size], -epsilon, epsilon)
            # Ensure the rest of the image is not perturbed
            mask = torch.ones_like(delta[i])
            mask[:, y:y+patch_size, x:x+patch_size] = 0
            delta.data[i] = delta.data[i] * (1 - mask)
        # Clamp the adversarial image to valid range
        delta.data = torch.clamp(images + delta.data, 0, 1) - images
        delta.grad.zero_()

    adv_images = torch.clamp(images + delta.detach(), 0, 1)
    return adv_images, patch_coords, target_labels

def create_patch_adversarial_dataset(model, dataloader, class_map, patch_size=32, epsilon=0.3, alpha=0.03, num_steps=20, output_dir="AdversarialTestSet3"):
    """
    Create adversarial dataset using patch-based targeted PGD attack
    """
    print(f"\nCreating patch-based adversarial dataset (patch size {patch_size}, epsilon {epsilon})...")
    model.eval()
    all_perturbed_images = []
    all_patch_coords = []
    all_target_labels = []
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
        images, labels = images.to(device), labels.to(device)
        imagenet_labels = torch.tensor([class_map[label.item()] for label in labels], device=device)
        adv_images, patch_coords, target_labels = targeted_pgd_patch_attack(
            model, images, imagenet_labels, patch_size=patch_size, epsilon=epsilon, alpha=alpha, num_steps=num_steps, num_classes=100)
        all_perturbed_images.append(adv_images.cpu())
        all_patch_coords.extend(patch_coords)
        all_target_labels.extend(target_labels.cpu().tolist())
    all_perturbed_images = torch.cat(all_perturbed_images, dim=0)
    save_adversarial_images(all_perturbed_images, output_dir)
    print(f"Saved patch-based adversarial images to {output_dir}")
    return all_perturbed_images, all_patch_coords, all_target_labels

def visualize_patch_attack(original_images, perturbed_images, patch_coords, labels, target_labels, predictions, label_map, num_examples=5):
    """
    Visualize original and patch-perturbed adversarial examples
    """
    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 5*num_examples))
    for i in range(num_examples):
        # Original image
        orig_img = original_images[i].cpu().numpy().transpose(1, 2, 0)
        orig_img = np.clip(orig_img * STD_NORMS + MEAN_NORMS, 0, 1)
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f'Original\nTrue: {label_map[labels[i]]}')
        axes[i, 0].axis('off')
        # Perturbed image
        pert_img = perturbed_images[i].cpu().numpy().transpose(1, 2, 0)
        pert_img = np.clip(pert_img * STD_NORMS + MEAN_NORMS, 0, 1)
        x, y = patch_coords[i]
        # Draw a rectangle around the patch
        import matplotlib.patches as patches
        axes[i, 1].imshow(pert_img)
        rect = patches.Rectangle((x, y), 32, 32, linewidth=2, edgecolor='r', facecolor='none')
        axes[i, 1].add_patch(rect)
        axes[i, 1].set_title(f'Patch-PGD\nTarget: {label_map[target_labels[i]]}\nPred: {label_map[predictions[i]]}')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()

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
    
    # Task 4: Adversarial Training
    print("\nTask 4: Implementing Adversarial Training...")
    
    # Create a new model for adversarial training
    adv_model = load_model()
    adv_model.train()  # Set to training mode
    
    # Training parameters
    num_epochs = 5
    learning_rate = 0.0001
    epsilon = 0.02
    alpha = 0.002
    num_steps = 10
    
    print(f"\nStarting adversarial training with parameters:")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epsilon: {epsilon}")
    print(f"Alpha: {alpha}")
    print(f"PGD steps: {num_steps}")
    
    # Train the model
    adv_model, train_losses = train_adversarial_model(
        adv_model, dataloader, num_epochs, learning_rate, epsilon, alpha, num_steps
    )
    
    # Plot training curves
    plot_training_curves(train_losses)
    
    # Collect metrics for different attacks
    metrics = {}
    
    print("\nCollecting metrics for different attacks...")
    
    # On clean images
    metrics['clean'] = collect_metrics(adv_model, dataloader, label_map, class_map, "clean")
    
    # On FGSM adversarial examples
    metrics['fgsm'] = collect_metrics(adv_model, adversarial_dataloader, label_map, class_map, "FGSM")
    
    # On PGD adversarial examples
    metrics['pgd'] = collect_metrics(adv_model, pgd_dataloader, label_map, class_map, "PGD")
    
    # On I-FGSM adversarial examples
    metrics['ifgsm'] = collect_metrics(adv_model, ifgsm_dataloader, label_map, class_map, "I-FGSM")
    
    # Generate visualizations
    plot_accuracy_comparison(metrics)
    plot_confidence_distribution(metrics)
    
    # Save metrics to file
    save_metrics_to_file(metrics)
    
    # Print results
    print("\nAdversarial Training Results:")
    print("===========================")
    for attack_type, attack_metrics in metrics.items():
        print(f"\n{attack_type.upper()} Attack:")
        print(f"Accuracy: {attack_metrics['accuracy']:.2f}%")
        print(f"Average Confidence: {attack_metrics['avg_confidence']:.4f}")
    
    # Save the adversarially trained model
    torch.save(adv_model.state_dict(), 'adversarially_trained_model.pth')
    print("\nAdversarially trained model saved as 'adversarially_trained_model.pth'")

    # Task 4 (part 2): Patch-based targeted PGD attack
    print("\nTask 4 (part 2): Patch-based targeted PGD attack...")
    patch_size = 32
    patch_epsilon = 0.3
    patch_alpha = 0.03
    patch_steps = 20
    patch_adv_images, patch_coords, patch_targets = create_patch_adversarial_dataset(
        model, dataloader, class_map, patch_size=patch_size, epsilon=patch_epsilon, alpha=patch_alpha, num_steps=patch_steps, output_dir="AdversarialTestSet3")
    # Evaluate model on patch adversarial dataset
    patch_dataset = torch.utils.data.TensorDataset(
        patch_adv_images.to(device),
        torch.tensor([reverse_class_map[class_map[label]] for label in dataset.targets], device=device)
    )
    patch_dataloader = torch.utils.data.DataLoader(patch_dataset, batch_size=32, shuffle=False)
    print("\nEvaluating model on patch-based adversarial dataset...")
    patch_top1_acc, patch_top5_acc = evaluate_model(model, patch_dataloader, label_map, class_map)
    print(f"Patch-PGD Adversarial Top-1 Accuracy: {patch_top1_acc:.2f}%")
    print(f"Patch-PGD Adversarial Top-5 Accuracy: {patch_top5_acc:.2f}%")
    # Visualize 5 examples
    with torch.no_grad():
        outputs = model(patch_adv_images[:5].to(device))
        _, predictions = outputs.max(1)
    visualize_patch_attack(
        original_images=dataset.data[:5] if hasattr(dataset, 'data') else torch.stack([dataset[i][0] for i in range(5)]),
        perturbed_images=patch_adv_images[:5],
        patch_coords=patch_coords[:5],
        labels=[class_map[label] for label in dataset.targets[:5]],
        target_labels=patch_targets[:5],
        predictions=predictions.cpu().numpy(),
        label_map=label_map,
        num_examples=5
    )

if __name__ == "__main__":
    main() 