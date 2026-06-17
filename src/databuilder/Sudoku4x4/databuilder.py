import torch
import torchvision
import numpy as np
from src.samplers.sudoku4x4_sampler import Sampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sampler = Sampler()

def load_mnist_by_digit():

    mnist_train = torchvision.datasets.MNIST(
        "src/datasets/",
        train=True, download=True,
        transform=torchvision.transforms.ToTensor()
    )

    mnist_test = torchvision.datasets.MNIST(
        "src/datasets/",
        train=False, download=True,
        transform=torchvision.transforms.ToTensor()
    )

    mnist_train_data = mnist_train.data.to(torch.float32) / 255.0    
    mnist_train_targets = mnist_train.targets
    mnist_test_data = mnist_test.data.to(torch.float32) / 255.0
    mnist_test_targets = mnist_test.targets

    train_ones_tensor = torch.where(mnist_train.targets == 1)[0]
    train_twos_tensor = torch.where(mnist_train.targets == 2)[0]
    train_threes_tensor = torch.where(mnist_train.targets == 3)[0]
    train_fours_tensor = torch.where(mnist_train.targets == 4)[0]

    test_ones_tensor = torch.where(mnist_test.targets == 1)[0]
    test_twos_tensor = torch.where(mnist_test.targets == 2)[0]
    test_threes_tensor = torch.where(mnist_test.targets == 3)[0]
    test_fours_tensor = torch.where(mnist_test.targets == 4)[0]

    min_train = min(len(train_ones_tensor), len(train_twos_tensor), len(train_threes_tensor), len(train_fours_tensor))
    min_test = min(len(test_ones_tensor), len(test_twos_tensor), len(test_threes_tensor), len(test_fours_tensor))

    by_digit_train = torch.stack([
        mnist_train_data[train_ones_tensor][:min_train],
        mnist_train_data[train_twos_tensor][:min_train],
        mnist_train_data[train_threes_tensor][:min_train],
        mnist_train_data[train_fours_tensor][:min_train]
    ]).unsqueeze(2) #[4, L, 1, 28, 28]

    by_digit_test = torch.stack([
        mnist_test_data[test_ones_tensor][:min_test],
        mnist_test_data[test_twos_tensor][:min_test],
        mnist_test_data[test_threes_tensor][:min_test],
        mnist_test_data[test_fours_tensor][:min_test]
    ]).unsqueeze(2) #[4, L, 1, 28, 28]

    return by_digit_train, by_digit_test

def get_mnist_digit_board(board, by_digit):
    B, N, _ = board.shape #[B, 4, 4]
    L = by_digit.shape[1]
    device = by_digit.device
    
    board = board.to(device)
    imgs = torch.zeros((B, 4, 4, 1, 28, 28), dtype=torch.float32, device=device) #[B, 4, 4, 1, 28, 28]
    for i in range(4):
        for j in range(4):
            indices = torch.randint(0, L, (B,), device=device)  #[B]
            digit = board[:, i, j] - 1  #[B]
            imgs[:, i, j] = by_digit[digit, indices]  #[B, 1, 28, 28]
    
    return imgs

def get_anchor_mnist_digit(by_digit):
    L = by_digit.shape[1]
    indices = torch.randint(0, L, (4,))  #[4]
    anchor_imgs = by_digit[torch.arange(4), indices]  #[4, 1, 28, 28]
    return anchor_imgs

def tensorized_split_boards(device, n_train, n_test):
    sat_train_boards = sampler.tensor_sat_sample_batch(device, n_train // 2).squeeze(1) #[n_train, 4, 4]
    sat_train_labels = torch.ones(n_train // 2, dtype=torch.long) #[n_train // 2]
    sat_test_boards = sampler.tensor_sat_sample_batch(device, n_test // 2).squeeze(1)
    sat_test_labels = torch.ones(n_test // 2, dtype=torch.long) #[n_test // 2]

    assert sampler.tensor_check(sat_train_boards).any().item() == True
    assert sampler.tensor_check(sat_test_boards).any().item() == True

    unsat_train_boards = sampler.tensor_unsat_sample_batch(device, n_train // 2).squeeze(1)
    unsat_test_boards = sampler.tensor_unsat_sample_batch(device, n_test // 2).squeeze(1)
    unsat_train_labels = torch.zeros(n_train // 2, dtype=torch.long) #[n_train // 2]
    unsat_test_labels = torch.zeros(n_test // 2, dtype=torch.long) #[n_test // 2]

    assert sampler.tensor_check(unsat_train_boards).any().item() != True
    assert sampler.tensor_check(unsat_test_boards).any().item() != True

    train_boards = torch.cat([sat_train_boards, unsat_train_boards], dim=0) #[n_train, 4, 4]
    train_labels = torch.cat([sat_train_labels, unsat_train_labels], dim=0) #[n_train]
    test_boards = torch.cat([sat_test_boards, unsat_test_boards], dim=0) #[n_test, 4, 4]
    test_labels = torch.cat([sat_test_labels, unsat_test_labels], dim=0) #[n_test]

    return train_boards, train_labels, test_boards, test_labels

def tensorized_split_boards_onlysat(device, n_train, n_test):
    train_boards = sampler.tensor_sat_sample_batch(device, n_train).squeeze(1) #[n_train, 4, 4]
    train_labels = torch.ones(n_train, dtype=torch.long) #[n_train]
    test_boards = sampler.tensor_sat_sample_batch(device, n_test).squeeze(1)
    test_labels = torch.ones(n_test, dtype=torch.long) #[n_test]

    return train_boards, train_labels, test_boards, test_labels

def build_dataset(train_boards, test_boards):
    by_digit_train, by_digit_test = load_mnist_by_digit()

    anchor_digits = get_anchor_mnist_digit(by_digit_train)

    train_imgs = get_mnist_digit_board(train_boards, by_digit_train)
    test_imgs = get_mnist_digit_board(test_boards, by_digit_test)

    return train_imgs, test_imgs, anchor_digits

def get_mnist_sudoku4x4_dataloader(device, n_train=100, n_test=25, batch_size=64):

    train_boards, train_labels, test_boards, test_labels = tensorized_split_boards(device, n_train, n_test)
    osat_train_boards, osat_train_labels, osat_test_boards, osat_test_labels = tensorized_split_boards_onlysat(device, n_train, n_test)

    train_imgs, test_imgs, anchor_digits = build_dataset(train_boards, test_boards)
    osat_train_imgs, osat_test_imgs, _ = build_dataset(osat_train_boards, osat_test_boards)

    train_idx = torch.arange(n_train)
    test_idx = torch.arange(n_test)


    train_set = torch.utils.data.TensorDataset(train_imgs, train_labels, train_boards, train_idx)
    test_set = torch.utils.data.TensorDataset(test_imgs, test_labels, test_boards, test_idx)

    osat_train_set = torch.utils.data.TensorDataset(osat_train_imgs, osat_train_labels, osat_train_boards, train_idx)
    osat_test_set = torch.utils.data.TensorDataset(osat_test_imgs, osat_test_labels, osat_test_boards, test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    osat_train_loader = torch.utils.data.DataLoader(
        osat_train_set, batch_size=batch_size, shuffle=True
    )
    osat_test_loader = torch.utils.data.DataLoader(
        osat_test_set, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, osat_train_loader, osat_test_loader, anchor_digits

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_loader, test_loader, osat_train_loader, osat_test_loader, anchor_digits = get_mnist_sudoku4x4_dataloader(device, n_train=100, n_test=1000, batch_size=64)
    print(len(test_loader.dataset)) #type:ignore
    imgs, labels, boards, idx = next(iter(train_loader))
    print(imgs.shape)

