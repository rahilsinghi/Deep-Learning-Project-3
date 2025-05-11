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

For detailed results and analysis, refer to the following files:

- **Task-by-Task Summary**: `task_summary.txt` provides a comprehensive overview of each task, including code explanations, results, and visualizations.
- **Visualizations**:
  - `accuracy_comparison_all_attacks.png`: Comparison of all attacks.
  - `attack_effectiveness.png`: Attack effectiveness (accuracy reduction).
  - `patch_attack_examples.png`: Examples of patch attacks.
  - `attack_parameters.png`: Comparison of attack parameters.
  - `transfer_attack_accuracy.png`: Impact of adversarial attacks on DenseNet-121.

## Suggestions for Further Reading

- **Adversarial Attacks**: Explore more advanced attacks and defenses in the field of adversarial machine learning.
- **Model Robustness**: Investigate techniques to improve model robustness against adversarial attacks.
- **Transferability**: Study the transferability of adversarial examples across different models and architectures.

For further details, see the code and visualizations in this repository.

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

## Task 5: Transferring Attacks

### Results and Analysis

| Dataset      | Top-1 Accuracy | Top-5 Accuracy |
|--------------|:--------------:|:--------------:|
| Original     | 74.80%         | 93.80%         |
| FGSM         | 28.20%         | 52.40%         |
| PGD          | 3.00%          | 15.20%         |
| Patch-PGD    | 32.60%         | 60.00%         |

### Discussion & Analysis

#### 1. **Transferability of Adversarial Attacks**
- **FGSM and PGD attacks** generated on ResNet-34 significantly reduce DenseNet-121's accuracy, even though the adversarial examples were not crafted for this model. This demonstrates the high transferability of adversarial examples between different deep networks.
- **PGD** remains the most effective, dropping Top-1 accuracy to just 3.00% and Top-5 to 15.20%.
- **Patch-PGD** (localized patch attack) is less effective than full-image attacks but still reduces accuracy substantially, showing that even small, localized perturbations can transfer.

#### 2. **Trends Observed**
- **Original images**: DenseNet-121 achieves high accuracy, as expected.
- **All adversarial sets**: There is a dramatic drop in both Top-1 and Top-5 accuracy, with PGD being the most severe.
- **Patch-based attacks**: While less effective than full-image attacks, they still cause a significant drop, especially in Top-5 accuracy.

#### 3. **Lessons Learned**
- **Adversarial examples are highly transferable** between different architectures, even when the attack is not tailored to the target model.
- **Localized attacks** (patch-based) are a real threat, as they can fool models even with limited perturbation area.

#### 4. **Mitigation Strategies**
- **Adversarial training** with a variety of attacks (including patch-based and transferred attacks) can help improve robustness.
- **Ensemble methods** and input preprocessing may reduce transferability, but are not foolproof.
- **Detection mechanisms** for adversarial examples are an active area of research.

All results are visualized in `transfer_attack_accuracy.png`.

For further details, see the code and visualizations in this repository. 