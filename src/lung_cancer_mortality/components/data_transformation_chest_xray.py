import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image

# Transformations applied: Resizing, random flipping, normalization
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def save_transformed_images(dataloader, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, (inputs, labels) in enumerate(dataloader):
        for j in range(inputs.size(0)):
            img = inputs[j]
            label = labels[j].item()
            class_dir = os.path.join(save_dir, str(label))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            img = img.numpy().transpose(1, 2, 0)
            img = (img * 255).astype('uint8')
            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(class_dir, f"img_{i * dataloader.batch_size + j}.png"))

def main():
    # Load data
    data_dir = 'C:/Users/angad/OneDrive/Desktop/lung_disease_project_main/lung-risk-predictor-vit/data/chest_xray_dataset'
    save_dir = 'C:/Users/angad/OneDrive/Desktop/lung_disease_project_main/lung-risk-predictor-vit/data/chest_xray_dataset/preprocessed_data'

    image_datasets = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    dataloaders = DataLoader(image_datasets, batch_size=32, shuffle=True, num_workers=4)

    # Save transformed images
    save_transformed_images(dataloaders, save_dir)

    class_names = image_datasets.classes

    # Example of using the dataloaders
    # Iterate through a batch of training data
    inputs, labels = next(iter(dataloaders))
    print(f"Inputs shape: {inputs.shape}")
    print(f"Labels shape: {labels.shape}")

if __name__ == '__main__':
    main()

