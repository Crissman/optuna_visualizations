import os
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import utils
import argparse

parser = argparse.ArgumentParser(description='MLP args')
parser.add_argument('-e', help='Number of epochs', default=20, type=int)
parser.add_argument('-t', help='Number of trials', default=100, type=int)
parser.add_argument('-nf', help="Don't use fashion MNIST", default=False, action='store_true')
parser.add_argument('-hp', help="Number of HPs to be tunes", default=0, type=int)
parser.add_argument('-suf', help="Suffix to add to end of trial name", default="")
args = parser.parse_args()

DEVICE = torch.device('cpu')
CLASSES = 10
# DIR = os.getcwd()
DIR = '../optuna'
EPOCHS = args.e

def define_model(trial):
    layers = []
    in_features = 28 * 28
    for i in range(2):
        out_features = trial.suggest_int('n_units_l{}'.format(i), 4, 128)
        activation = trial.suggest_categorical(
            'activation_l{}'.format(i), choices=['relu', 'tanh'])
        p = trial.suggest_uniform('dropout_l{}'.format(i), 0.2, 0.5)
        layers.append(nn.Linear(in_features, out_features))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        else:
            assert False
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*layers)

def get_mnist(trial):
    batch_size = trial.suggest_categorical(
        'batch_size', choices=[8, 16, 32, 64, 128])
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            DIR, train=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, test_loader

def objective(trial):
    model = define_model(trial).to(DEVICE)
    optimizer_name = trial.suggest_categorical(
        'optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    train_loader, test_loader = get_mnist(trial)
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data
            loss.backward()
            optimizer.step()
        epoch_loss /= len(train_loader.dataset)
        print('Epoch {}: {}'.format(epoch, epoch_loss))
        trial.report(epoch_loss, step=epoch)
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy

if __name__ == '__main__':
    mnist_type=('mnist' if args.nf else 'fash_mnist')
    study = optuna.create_study(
        # sampler=optuna.samplers.RandomSampler(),
        sampler=optuna.samplers.TPESampler(),
        study_name=mnist_type+'-hp_all-e'+str(args.e)+'-t'+str(args.t)+args.suf,
        storage='sqlite:///nnTPE.db',
        load_if_exists=True,
        direction='maximize',
    )
    if args.hp:
        fixed_params = {}
        all_params = list(optuna.importance.get_param_importances(study).keys())
        print('Optimizing for values of ', all_params[:args.hp])
        fix_hps = list(optuna.importance.get_param_importances(study).keys())[args.hp:]
        print('Fixing hyperparamteter values of ', fix_hps)
        while len(fix_hps):
            param = fix_hps.pop()
            fixed_params[param] = study.best_params[param]

        partial_sampler = optuna.samplers.PartialFixedSampler(fixed_params, study.sampler)
        study.sampler = partial_sampler
            
    study.optimize(objective, n_trials=args.t)
