import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision

from src.databuilder.Kandlogic.databuilder import get_kandlogic_dataloader, get_anchors

"""Quick somoke test and datapoint visualization for Kandlogic dataset"""

def visualize_kandlogic_datapoint(images, labels, concepts):
    images_flat = images.view(-1, 3, 28, 28)  # [9, 3, 28, 28]
    concept = concepts
    label = labels.item()

    print(f"Concept: {concept}")
    print(f"Label: {label}")

    print(f"Images shape: {images_flat.shape}")

    grid_img = torchvision.utils.make_grid(images_flat, nrow=3)

    img = grid_img.permute(1, 2, 0).numpy() * 255
    img = Image.fromarray(img.astype('uint8'))
    img.show()

if __name__ == "__main__":
    train_loader, test_loader, anchor_images, anchor_labels = get_kandlogic_dataloader(b_size=64)
    print(f"Number of training samples: {len(train_loader.dataset)}") #type:ignore
    print(f"Number of testing samples: {len(test_loader.dataset)}")   #type:ignore

    # Visualize a random datapoint from the training set
    random_idx = torch.randint(0, len(test_loader.dataset), (1,)).item() #type:ignore
    images, labels, concepts, _ = test_loader.dataset[random_idx]
    visualize_kandlogic_datapoint(images, labels, concepts)
