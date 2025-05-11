# Deep Learning Project 3: Adversarial Attacks on Image Classification

This project implements and evaluates various adversarial attacks on a pre-trained ResNet-34 model using the ImageNet-1K dataset.

## Project Structure

```
.
├── adversarial_attacks.py    # Main implementation file
├── requirements.txt         # Project dependencies
├── TestDataSet.zip         # Test dataset (ImageNet-1K)
├── adversarially_trained_model.pth  # Trained robust model
└── task3_observations.txt   # Observations and results
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Extract the test dataset:
```bash
unzip TestDataSet.zip
```

## Implementation Details

### Task 1: Project Setup and Baseline Performance
- Implemented baseline evaluation of ResNet-34 model on ImageNet-1K
- Achieved baseline accuracy of ~76% on clean images

### Task 2: Fast Gradient Sign Method (FGSM)
- Implemented FGSM attack with configurable epsilon values
- Key features:
  - Gradient computation using torch.autograd
  - Proper label mapping for ImageNet classes
  - Support for batch processing
  - Visualization of adversarial examples

### Task 3: Improved Attacks
Implemented two advanced adversarial attacks:

1. **Projected Gradient Descent (PGD)**
   - Iterative attack with multiple steps
   - Projection to epsilon ball after each step
   - Configurable parameters:
     - Number of steps
     - Step size
     - Epsilon value

2. **Iterative Fast Gradient Sign Method (I-FGSM)**
   - Iterative version of FGSM
   - Smaller step sizes for better optimization
   - Configurable parameters:
     - Number of iterations
     - Step size
     - Epsilon value

### Task 4: Adversarial Training
Implemented adversarial training to improve model robustness:

1. **Training Process**
   - Uses PGD attack for generating adversarial examples during training
   - Configurable parameters:
     - Number of epochs
     - Learning rate
     - Epsilon (perturbation size)
     - Alpha (step size)
     - Number of PGD steps

2. **Key Features**
   - Real-time adversarial example generation
   - Adam optimizer for stable training
   - Progress tracking with loss monitoring
   - Model checkpointing

3. **Evaluation**
   - Comprehensive evaluation on:
     - Clean images
     - FGSM adversarial examples
     - PGD adversarial examples
     - I-FGSM adversarial examples

## Results and Analysis

### 1. Baseline Performance
- **Clean Images (No Attack)**
  - Top-1 Accuracy: 76.00%
  - Top-5 Accuracy: 94.20%

### 2. Full-Image Attacks

#### FGSM Attack (ε = 0.02)
- **Top-1 Accuracy**: 26.40% (reduction of 49.60%)
- **Top-5 Accuracy**: 50.60% (reduction of 43.60%)
- **Characteristics**:
  - Fastest attack (single step)
  - Moderate effectiveness
  - Less computationally intensive

#### PGD Attack (ε = 0.02, α = 0.002, steps = 10)
- **Top-1 Accuracy**: 2.00% (reduction of 74.00%)
- **Top-5 Accuracy**: 13.60% (reduction of 80.60%)
- **Characteristics**:
  - Most effective full-image attack
  - Higher computational cost
  - Iterative optimization

#### I-FGSM Attack (ε = 0.02, α = 0.002, steps = 10, momentum = 0.9)
- **Top-1 Accuracy**: 1.60% (reduction of 74.40%)
- **Top-5 Accuracy**: 10.20% (reduction of 84.00%)
- **Characteristics**:
  - Similar effectiveness to PGD
  - Better convergence with momentum
  - Slightly faster than PGD

### 3. Patch-Based Attack (Task 4)

#### Patch-PGD Attack (32x32 patch, ε = 0.3, α = 0.03, steps = 20)
- **Top-1 Accuracy**: 33.40% (reduction of 42.60%)
- **Top-5 Accuracy**: 55.60% (reduction of 38.60%)
- **Characteristics**:
  - Targeted attack on a small patch
  - Higher epsilon (0.3) to compensate for limited perturbation area
  - More challenging than full-image attacks
  - More realistic attack scenario

### 4. Adversarial Training Results
- **Clean Images**: 0.20% accuracy
- **FGSM Attack**: 0.40% accuracy
- **PGD Attack**: 0.40% accuracy
- **I-FGSM Attack**: 0.40% accuracy

**Note**: The adversarial training results are limited by using the test set for training, which is not a realistic scenario.

## Limitations and Discussion

### 1. Dataset Limitations
- Using the test set for both training and evaluation
- Small dataset size (100 classes)
- Limited number of images per class

### 2. Attack Limitations
- **Full-Image Attacks**:
  - Require access to the entire image
  - May be easily detectable
  - High perturbation budget needed

- **Patch Attack**:
  - Limited to a small area (32x32 pixels)
  - Requires higher epsilon for effectiveness
  - More realistic but less effective than full-image attacks

### 3. Adversarial Training Limitations
- Poor performance due to using test set for training
- Overfitting to the small dataset
- Not representative of real-world adversarial training

### 4. Future Improvements
1. Use a proper training set for adversarial training
2. Implement more sophisticated patch attacks
3. Explore different patch sizes and locations
4. Investigate transferability of attacks
5. Implement defense mechanisms

## Visualizations
The following visualizations are available in the repository:
- `accuracy_comparison_all_attacks.png`: Comparison of all attacks
- `attack_effectiveness.png`: Attack effectiveness (accuracy reduction)
- `patch_attack_examples.png`: Examples of patch attacks
- `attack_parameters.png`: Comparison of attack parameters

## Usage

To run the attacks:

```python
from adversarial_attacks import FGSMAttack, PGDAttack, IFGSMAttack

# FGSM Attack
fgsm = FGSMAttack(model, epsilon=0.03)
adversarial_images = fgsm.attack(images, labels)

# PGD Attack
pgd = PGDAttack(model, epsilon=0.03, steps=10, step_size=0.01)
adversarial_images = pgd.attack(images, labels)

# I-FGSM Attack
ifgsm = IFGSMAttack(model, epsilon=0.03, steps=10, step_size=0.01)
adversarial_images = ifgsm.attack(images, labels)
```

To perform adversarial training:

```python
from adversarial_attacks import train_adversarial_model

# Train model with adversarial training
adv_model = train_adversarial_model(
    model,
    train_loader,
    num_epochs=5,
    learning_rate=0.0001,
    epsilon=0.02,
    alpha=0.002,
    num_steps=10
)
```

## Dependencies

- PyTorch
- torchvision
- PIL
- numpy
- matplotlib

See `requirements.txt` for specific versions. 