import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Constants
MEAN_NORMS = np.array([0.485, 0.456, 0.406])
STD_NORMS = np.array([0.229, 0.224, 0.225])
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATASETS = {
    "Original": "TestDataSet",
    "FGSM": "AdversarialTestSet1",
    "PGD": "AdversarialTestSet_PGD",
    "Patch-PGD": "AdversarialTestSet3"
}

# Load label and class mapping
with open('TestDataSet/labels_list.json', 'r') as f:
    labels = json.load(f)
label_map = {int(label.split(': ')[0]): label.split(': ')[1] for label in labels}
class_map = {i: i + 401 for i in range(100)}
reverse_class_map = {v: k for k, v in class_map.items()}

# Preprocessing
plain_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_NORMS, std=STD_NORMS)
])

# Load DenseNet-121
print("Loading DenseNet-121 (ImageNet-1K pre-trained)...")
densenet = torchvision.models.densenet121(weights='IMAGENET1K_V1').to(DEVICE)
densenet.eval()

# Evaluation function
def evaluate_model(model, dataloader, label_map, class_map):
    correct_1 = 0
    correct_5 = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(DEVICE)
            outputs = model(images)
            imagenet_labels = torch.tensor([class_map[label.item()] for label in labels], device=DEVICE)
            # Top-1
            _, predicted = outputs.max(1)
            correct_1 += (predicted == imagenet_labels).sum().item()
            # Top-5
            _, predicted_5 = outputs.topk(5, 1, largest=True, sorted=True)
            imagenet_labels = imagenet_labels.view(imagenet_labels.size(0), -1).expand_as(predicted_5)
            correct_5 += predicted_5.eq(imagenet_labels).sum().item()
            total += images.size(0)
    top1 = 100. * correct_1 / total
    top5 = 100. * correct_5 / total
    return top1, top5

# Helper to load adversarial datasets
class AdvTensorDataset(torch.utils.data.Dataset):
    def __init__(self, adv_dir, targets):
        self.adv_dir = adv_dir
        self.targets = targets
        self.img_files = [f"adv_image_{i}.jpg" for i in range(len(targets))]
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        img_path = os.path.join(self.adv_dir, self.img_files[idx])
        img = Image.open(img_path).convert('RGB')
        img = plain_transforms(img)
        label = self.targets[idx]
        return img, label

# Get original dataset targets
orig_dataset = torchvision.datasets.ImageFolder(DATASETS["Original"], transform=plain_transforms)
orig_targets = [sample[1] for sample in orig_dataset.samples]

# Prepare dataloaders for all datasets
loaders = {}
# Original
loaders["Original"] = torch.utils.data.DataLoader(orig_dataset, batch_size=BATCH_SIZE, shuffle=False)
# Adversarial sets
for name in ["FGSM", "PGD", "Patch-PGD"]:
    loaders[name] = torch.utils.data.DataLoader(
        AdvTensorDataset(DATASETS[name], orig_targets),
        batch_size=BATCH_SIZE, shuffle=False)

# Evaluate and log results
results = {}
for name, loader in loaders.items():
    print(f"\nEvaluating DenseNet-121 on {name} dataset...")
    top1, top5 = evaluate_model(densenet, loader, label_map, class_map)
    results[name] = {"top1": top1, "top5": top5}
    print(f"{name} Top-1 Accuracy: {top1:.2f}%")
    print(f"{name} Top-5 Accuracy: {top5:.2f}%")

# Save results to file
with open("transfer_attack_results.txt", "w") as f:
    f.write("DenseNet-121 Transfer Attack Evaluation\n")
    f.write("====================================\n\n")
    for name, res in results.items():
        f.write(f"{name} Set:\n")
        f.write(f"  Top-1 Accuracy: {res['top1']:.2f}%\n")
        f.write(f"  Top-5 Accuracy: {res['top5']:.2f}%\n\n")

# Visualization
names = list(results.keys())
top1s = [results[n]["top1"] for n in names]
top5s = [results[n]["top5"] for n in names]

x = np.arange(len(names))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, top1s, width, label='Top-1 Accuracy')
rects2 = ax.bar(x + width/2, top5s, width, label='Top-5 Accuracy')
ax.set_ylabel('Accuracy (%)')
ax.set_title('DenseNet-121 Accuracy on Original and Adversarial Test Sets')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()
for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')
plt.tight_layout()
plt.savefig('transfer_attack_accuracy.png')
plt.close()

print("\nResults saved to transfer_attack_results.txt and transfer_attack_accuracy.png.") 