import torch
import torch.nn as nn
import torchvision 
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
from model import build_Vit
from tqdm import tqdm

batch_size = 4
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_ds(batch_size):
    transform_training_data = Compose(
        # https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py for normalize values
        [RandomCrop(32, padding=4), Resize((224)), RandomHorizontalFlip(), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_data = torchvision.datasets.CIFAR10(
        root='/home/mahdi/git/ViT/data', train=True, download=True, transform=transform_training_data)

    test_data = torchvision.datasets.CIFAR10(
        root='/home/mahdi/git/ViT/data', train=False, download=True, transform=transform_training_data)

    return torch.utils.data.DataLoader(test_data), torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True)

def eval(test_data, model) -> None:
    model.eval()
    acc_total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_data):

            preds = model(imgs)
            pred_cls = preds.data.max(1)[1]
            acc_total += pred_cls.eq(labels.data).cpu().sum()

    acc = acc_total / len(test_data.dataset)
    print('Accuracy on test set = '+str(acc))
    return


def train(train_data, epochs, batch_size, N, L, C, D, h, dropout):
    model = build_Vit(N, L, C, D, h, D, dropout, 224, 224, 3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=10e-3, weight_decay=0.03)
    sched = torch.optim.lr_scheduler.LinearLR(optimizer)
    model.train()
    for epoch in tqdm(range(epochs), total=epochs):
        for batch_idx, (input, label) in enumerate(tqdm(train_data)):
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            sched.step()
            
    return model 

if __name__ == '__main__':
    test_data, train_data = get_ds(batch_size)
    model = train(train_data, 30, 4, 196, 12, len(classes), 768, 12, 0.1)
    eval(test_data, model)
