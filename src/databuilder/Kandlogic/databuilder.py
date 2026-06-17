import torch

NUM_PRIMITIVES = 9
NUM_SUBFIGURES = 3

def get_anchors(train_imgs, train_concepts):
    """Return one canonical image per class (sorted by class label)."""
    anchor_labels, anchor_images = [], []
    seen = set()
    for i in range(len(train_concepts)):
        shape = train_concepts[i, 0, 0, 0].item()
        color = train_concepts[i, 0, 0, 1].item()
        lbl = shape*10 + color
        if lbl not in seen:
            seen.add(lbl)
            anchor_labels.append(torch.tensor([shape, color]))
            anchor_images.append(train_imgs[i, 0, 0])
        
        if len(anchor_labels) == 9:
            break

    anchor_images = torch.stack(anchor_images)
    anchor_labels = torch.stack(anchor_labels)
    return anchor_images, anchor_labels

def get_kandlogic_dataloader(b_size=64):
    train_data = torch.load('src/datasets/Kandlogic/kand-3k-train.pt', weights_only=False)
    test_data = torch.load('src/datasets/Kandlogic/kand-3k-test.pt', weights_only=False)
    N_train = len(train_data)
    N_test = len(test_data)

    train_concepts = []
    train_images = []
    train_labels = []

    test_concepts = []
    test_images = []
    test_labels = []

    for i in range(N_train):
        train_concepts.append(train_data[i]['concepts'])
        train_images.append(train_data[i]['images'])
        train_labels.append(train_data[i]['labels'])

    for i in range(N_test):
        test_concepts.append(test_data[i]['concepts'])
        test_images.append(test_data[i]['images'])
        test_labels.append(test_data[i]['labels'])

    train_concepts = torch.stack(train_concepts)
    train_concepts = train_concepts.permute(0, 1, 3, 2)

    train_images = torch.stack(train_images)
    train_labels = torch.tensor(train_labels)


    test_concepts = torch.stack(test_concepts)
    test_concepts = test_concepts.permute(0, 1, 3, 2)

    test_images = torch.stack(test_images)
    test_labels = torch.tensor(test_labels)

    anchor_images, anchor_labels = get_anchors(train_images, train_concepts)

    def _make_dataset(images, concepts, labels, train=False):
        idx = torch.arange(len(images))
        if train:
            return torch.utils.data.TensorDataset(
                images, labels, concepts, idx
            )
        else:
            return torch.utils.data.TensorDataset(
                images, labels, concepts, idx
            )
        
    train_set = _make_dataset(train_images, train_concepts, train_labels, train=True)
    test_set  = _make_dataset(test_images, test_concepts, test_labels)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=b_size, shuffle=True
    )

    test_loader  = torch.utils.data.DataLoader(
        test_set,  batch_size=b_size, shuffle=False
    )

    return train_loader, test_loader, anchor_images, anchor_labels

if __name__ == '__main__':
    train_loader, test_loader, anchor_images, anchor_labels = get_kandlogic_dataloader()
    images, labels, concepts, idx = next(iter(test_loader))

