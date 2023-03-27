""""Modified code from https://raw.githubusercontent.com/pytorch/examples/main/mnist/main.py"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os

# Imports required when adding distribution
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from types import SimpleNamespace


def init_dist():
    # Get information from torchrun
    size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialise the process group
    dist.init_process_group('nccl')

    return SimpleNamespace(
        rank=rank, size=size,
        local_size=local_size, local_rank=local_rank
    )


def cleanup():
    dist.destroy_process_group()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, test_sampler):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_sampler)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_sampler),
        100. * correct / len(test_sampler)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size per gpu for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size per gpu for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=16, metavar='N',
                        help='number of epochs to train (default: 16)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data-dir', default='./data',
                        help='Directory dataset is in')
    parser.add_argument('--output-dir', default='./output/default',
                        help='Specify where to save the current model and log')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Get information about distribution
    dist = init_dist()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dist.local_rank)
    device = torch.device('cuda:{}'.format(dist.local_rank))

    # Scale the learning rate by the global size
    args.lr = args.lr * dist.size

    torch.manual_seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        # remove shuffle from cuda_kwargs as it is specified in the sampler
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False,
                                  transform=transform)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs, sampler=test_sampler)

    model = Net().to(device)
    model = DDP(model, device_ids=[device])
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, test_sampler)
        scheduler.step()

    if args.save_model and dist.rank == 0:
        torch.save(model.state_dict(), "{}/mnist_cnn.pt".format(args.output_dir))

    cleanup()


if __name__ == '__main__':
    main()
