import os
from PIL import Image
import torchvision.transforms as transforms

# Transformations applied: Resizing, random flipping, normalization
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def apply_transform_and_save(input_dir, output_dir, transform):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
                img_path = os.path.join(root, file)
                img = Image.open(img_path).convert('RGB')
                transformed_img = transform(img)


                transformed_img = transforms.ToPILImage()(transformed_img)

                # Saving processed image
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                transformed_img.save(os.path.join(output_subdir, file))


input_dir = 'C:/Users/angad/OneDrive/Desktop/lung_disease_project_main/lung-risk-predictor-vit/data/NIH-dataset'
output_dir = 'C:/Users/angad/OneDrive/Desktop/lung_disease_project_main/lung-risk-predictor-vit/data/nih_processed_images'


apply_transform_and_save(input_dir, output_dir, transform)