import argparse
import numpy.random
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from timm.loss import LabelSmoothingCrossEntropy
from homura.vision.models.cifar_resnet import wrn28_2, wrn28_10, resnet20, resnet56, resnext29_32x4d
from asam_pyramid import ASAM, SAM
from pyramid_shakedrop import CutMixCrossEntropyLoss, CutMix, ImageNetPolicy, PyramidNet, CIFAR10Policy
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','robust-generalization-flatness'))
import attacks.weights
from copy import deepcopy
from collections import OrderedDict
parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', type=str, default='/mnt/SHARED/mmueller67/ASAM/snapshots/ASAM_original_eta_zero_3seeds/', help='directory to folder with saved models')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for SGD-PGD, if > num_examples, then more than one update step per epoch is made')
args = parser.parse_args()

sys.path.append('/mnt/SHARED/mmueller67')
from find_minima.CIFAR.models.wrn import WideResNet
from find_minima.utils.helpers import get_model_name



def equal_models(m1, m2):
    for (n1, p1), (n2,p2) in zip(m1.named_parameters(), m2.named_parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def cuda_to_cpu_statedict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def load_cifar(data_loader, batch_size=256, num_workers=2, autoaugment=False, data_path = '/scratch/datasets/CIFAR100/'):
    if data_loader == CIFAR10:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    # Transforms
    if autoaugment:
        raise NotImplementedError
    else:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # DataLoader
    train_set = data_loader(root=data_path, train=True, download=False, transform=train_transform)
    # if args.cutmix:
        # train_set = CutMix(train_set, num_class=100, beta=1.0, prob=0.5, num_mix=2)
    test_set = data_loader(root=data_path, train=False, download=False, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader

# load model


model_original = wrn28_10(num_classes=100)
print('Model created...')
model_name = 'CIFAR100_wrn28_10_ASAM_lr0.1_bsize128_epochs100_rho2.0_layerwiseFalse_filterwiseFalse_elementwiseTrue_autoaugmentFalse_cutmixFalse_m128_eta0.0_ElementWiseL2NormAsam_seed0_last.pt'
state_dict = torch.load(os.path.join(args.path, model_name), map_location=torch.device('cpu'))
state_dict = cuda_to_cpu_statedict(state_dict)
model_original.load_state_dict(state_dict)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_original.to(device)
# check if large loss of eval adversarial step is also obtained in train mode:
# -> No!
_, norm, p, elementwise, layerwise, filterwise, rho = ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 2.)
print(norm, rho)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
testloader, _ = load_cifar(CIFAR100, args.batch_size, autoaugment=False)

batch_losses_eval_eval_adv = []
batch_losses_eval_train_adv = []
batch_losses_train_eval_adv = []
batch_losses_train_train_adv = []

batch_losses_eval = []
batch_losses_train = []

for idx, (inputs, targets) in enumerate(testloader):
    inputs = inputs.to(device)
    targets = targets.type(torch.int64).to(device)
    print('Batch {}/{}'.format(idx, len(testloader)))
    model_evalmode = deepcopy(model_original)
    model_evalmode.eval()
    optimizer_evalmode = torch.optim.SGD(model_evalmode.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=1e-5)

    minimizer_evalmode = ASAM(optimizer_evalmode, model_evalmode, rho=rho, eta=0., layerwise=layerwise,
                             elementwise=elementwise, filterwise=filterwise, p=p, normalize_bias=False)
    batch_loss_evalmode = criterion(model_evalmode(inputs), targets)
    batch_loss_evalmode.mean().backward()
    batch_losses_eval.append(batch_loss_evalmode.cpu().mean().item())
    minimizer_evalmode.ascent_step()
    model_evalmode.eval()
    predictions = model_evalmode(inputs)
    batch_loss_evalmode_eval = criterion(predictions, targets)
    batch_losses_eval_eval_adv.append(batch_loss_evalmode_eval.cpu().mean().item())

    model_evalmode.train()
    predictions = model_evalmode(inputs)
    batch_loss_evalmode_train = criterion(predictions, targets)
    batch_losses_eval_train_adv.append(batch_loss_evalmode_train.cpu().mean().item())

    # equivalently, check if small loss of train adversarial step is also obtained in eval mode
    model_trainmode = deepcopy(model_original)
    model_trainmode.train()
    optimizer_trainmode = torch.optim.SGD(model_trainmode.parameters(), lr=0.1,
                                          momentum=0.9, weight_decay=1e-5)

    minimizer_trainmode = ASAM(optimizer_trainmode, model_trainmode, rho=rho, eta=0., layerwise=layerwise,
                               elementwise=elementwise, filterwise=filterwise, p=p, normalize_bias=False)
    batch_loss_trainmode = criterion(model_trainmode(inputs), targets)
    batch_loss_trainmode.mean().backward()
    batch_losses_train.append(batch_loss_trainmode.cpu().mean().item())
    minimizer_trainmode.ascent_step()

    model_trainmode.eval()
    predictions = model_trainmode(inputs)
    batch_loss_trainmode_eval = criterion(predictions, targets)
    batch_losses_train_eval_adv.append(batch_loss_trainmode_eval.cpu().mean().item())

    model_trainmode.train()
    predictions = model_trainmode(inputs)
    batch_loss_trainmode_train = criterion(predictions, targets)
    batch_losses_train_train_adv.append(batch_loss_trainmode_train.cpu().mean().item())



# write to csv
filename = args.path + 'train_vs_eval_{}.csv'.format(model_name)
file_exists = os.path.isfile(filename)

with open(filename, 'a') as f:
    if not file_exists:
        # write header
        f.write('eval_eval_adv,eval_train_adv,train_train_adv,train_eval_adv,eval,train' + '\n')
    for a,b,c,d,e,f_ in zip(batch_losses_eval_eval_adv, batch_losses_eval_train_adv, batch_losses_train_train_adv, batch_losses_train_eval_adv, batch_losses_eval, batch_losses_train):
        f.write('{}, {}, {}, {}, {}, {}'.format(a,b,c,d,e,f_)+ '\n')