import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision

from src.databuilder.MNIST_Evenodd.databuilder import get_mnist_evenodd_dataloader, get_mnist_addition_dataloader

"""Quick somoke test and datapoint visualization for MNIST_Evenodd dataset"""

def visualize_mnist_datapoint(image1, image2, label_addition, label1=None, label2=None, Train=True):
    if Train:
        print(f"Label Addition: {label_addition.item()}")
    else:
        print(f"Label 1: {label1.item()}") #type:ignore
        print(f"Label 2: {label2.item()}") #type:ignore
        print(f"Label Addition: {label_addition.item()}") #type:ignore

    grid_img = torchvision.utils.make_grid([image1, image2], nrow=2)

    img = grid_img.permute(1, 2, 0).numpy() * 255
    img = Image.fromarray(img.astype('uint8'))
    img.show()

if __name__ == "__main__":
    train_loader, test_loader, anchor_images = get_mnist_evenodd_dataloader(n_train=1000, n_test=200, b_size=64)
    print(f"Number of training samples: {len(train_loader.dataset)}") #type:ignore
    print(f"Number of testing samples: {len(test_loader.dataset)}")   #type:ignore

    """Denote that train set is for even/odd classification, while test set is for addition task. This is to test the generalization of the model from one task to another."""

    # Visualize a random datapoint from the training set
    random_idx = torch.randint(0, len(train_loader.dataset), (1,)).item() #type:ignore
    (image1, image2), label_addition, _ = train_loader.dataset[random_idx]
    visualize_mnist_datapoint(image1, image2, label_addition=label_addition, Train=True)

    # Visualize a random datapoint from the testing set
    random_idx = torch.randint(0, len(test_loader.dataset), (1,)).item() #type:ignore
    (image1, image2), label_addition, label1, label2, _ = test_loader.dataset[random_idx]
    visualize_mnist_datapoint(image1, image2, label_addition=label_addition, label1=label1, label2=label2, Train=False)