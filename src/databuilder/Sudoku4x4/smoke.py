import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision

from src.databuilder.Sudoku4x4.databuilder import get_mnist_sudoku4x4_dataloader

"""Quick somoke test and datapoint visualization for Sudoku4x4 dataset"""

def visualize_sudoku4x4_datapoint(images, label, board, Train=True):
    images_flat = images.view(-1, 1, 28, 28)

    print(f"Label: {label.item()}")
    print(f"Board: {board}")

    grid_img = torchvision.utils.make_grid(images_flat, nrow=4)

    img = grid_img.permute(1, 2, 0).numpy() * 255
    img = Image.fromarray(img.astype('uint8'))
    img.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, osat_train_loader, osat_test_loader, anchor_digits = get_mnist_sudoku4x4_dataloader(device, n_train=1000, n_test=200, batch_size=64)
    print(f"Number of training samples: {len(train_loader.dataset)}") #type:ignore
    print(f"Number of testing samples: {len(test_loader.dataset)}")   #type:ignore

    # Visualize a random datapoint from the training set
    random_idx = torch.randint(0, len(train_loader.dataset), (1,)).item() #type:ignore
    images, label, board, _ = train_loader.dataset[random_idx] #type:ignore
    visualize_sudoku4x4_datapoint(images, label, board, Train=True)