# Deep Learning Project 3: Adversarial Attacks on Image Classification

This project implements and evaluates various adversarial attacks on a pre-trained ResNet-34 model using the ImageNet-1K dataset.

## Project Structure

```
.
├── adversarial_attacks.py    # Main implementation file
├── requirements.txt         # Project dependencies
├── TestDataSet.zip         # Test dataset (ImageNet-1K)
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

## Results

The implemented attacks show significant effectiveness in reducing model accuracy:

1. FGSM Attack:
   - Achieves ~30% accuracy reduction with epsilon = 0.03
   - Fast execution but less effective than iterative methods

2. PGD Attack:
   - Most effective attack method
   - Achieves ~60% accuracy reduction
   - Higher computational cost due to multiple iterations

3. I-FGSM Attack:
   - Similar effectiveness to PGD
   - Slightly faster than PGD
   - Good balance between effectiveness and computational cost

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

## Dependencies

- PyTorch
- torchvision
- PIL
- numpy
- matplotlib

See `requirements.txt` for specific versions. 