import argparse
import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='baseline')
parser.add_argument('--act', default='relu', choices=['relu', 'tanh', 'sigmoid', 'leaky_relu'])
parser.add_argument('--init', default='default', choices=['default', 'xavier', 'kaiming'])
parser.add_argument('--opt', default='adam', choices=['adam', 'sgd', 'momentum', 'adagrad'])
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--data', default='data')
parser.add_argument('--logdir', default='runs')
args = parser.parse_args()

batch_size = 64
test_batch_size = 1000
epochs = args.epochs
lr = args.lr
try_cuda = True
seed = 1000
logging_interval = 10
logging_dir = os.path.join(args.logdir, args.name)

writer = SummaryWriter(logging_dir)

torch.manual_seed(seed)
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data, train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data, train=False, transform=transform),
    batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

activations = {'relu': F.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid,
               'leaky_relu': lambda x: F.leaky_relu(x, 0.01)}


class Net(nn.Module):
    def __init__(self, act):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.act = activations[act]
        self.record = None

    def forward(self, x):
        z1 = self.conv1(x)
        a1 = self.act(z1)
        p1 = F.max_pool2d(a1, 2)
        z2 = self.conv2(p1)
        a2 = self.act(z2)
        p2 = F.max_pool2d(a2, 2)
        z3 = self.fc1(p2.flatten(1))
        a3 = self.act(z3)
        x = self.fc2(self.drop(a3))
        if self.record is not None:
            self.record.update({
                'net_input/conv1': z1, 'activation/conv1': a1, 'maxpool/conv1': p1,
                'net_input/conv2': z2, 'activation/conv2': a2, 'maxpool/conv2': p2,
                'net_input/fc1': z3, 'activation/fc1': a3, 'net_input/fc2': x})
        x = F.softmax(x, dim=1)
        return x


model = Net(args.act)
if args.init != 'default':
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if args.init == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
if cuda:
    model.cuda()

if args.opt == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif args.opt == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr)
elif args.opt == 'momentum':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
else:
    optimizer = optim.Adagrad(model.parameters(), lr=lr)

eps = 1e-13
monitor_x = next(iter(test_loader))[0]
if cuda:
    monitor_x = monitor_x.cuda()


def log_tensor(tag, t, step):
    t = t.detach().float().cpu()
    writer.add_scalar(tag + '/min', t.min().item(), step)
    writer.add_scalar(tag + '/max', t.max().item(), step)
    writer.add_scalar(tag + '/mean', t.mean().item(), step)
    writer.add_scalar(tag + '/std', t.std().item(), step)
    writer.add_histogram(tag, t.numpy(), step)


def monitor(step):
    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        log_tensor('%s/%s' % ({'weight': 'weights', 'bias': 'biases'}[attr], layer), param, step)
    model.eval()
    model.record = {}
    with torch.no_grad():
        model(monitor_x)
    for tag, t in model.record.items():
        log_tensor(tag, t, step)
    model.record = None


def train(epoch):
    model.train()
    criterion = nn.NLLLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(torch.log(output + eps), target)
        loss.backward()
        optimizer.step()

        if batch_idx % logging_interval == 0:
            n_iter = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), n_iter)
            if batch_idx % (logging_interval * 30) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()), flush=True)

    monitor(epoch * len(train_loader))


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.NLLLoss(reduction='sum')

    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)

            test_loss += criterion(torch.log(output + eps), target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print('Test Epoch: {}\tAverage loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, test_loss, correct, len(test_loader.dataset), test_accuracy), flush=True)

    n_iter = epoch * len(train_loader)
    writer.add_scalar('test/loss', test_loss, n_iter)
    writer.add_scalar('test/accuracy', test_accuracy, n_iter)
    writer.add_scalar('test/error', 100. - test_accuracy, n_iter)
    return test_loss, test_accuracy


history = []
monitor(0)
start = time.time()
for epoch in range(1, epochs + 1):
    train(epoch)
    history.append(test(epoch))

writer.close()
with open(os.path.join(logging_dir, 'summary.json'), 'w') as f:
    json.dump({'args': vars(args), 'test': history, 'seconds': time.time() - start}, f)
