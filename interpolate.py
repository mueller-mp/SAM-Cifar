import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from timm.loss import LabelSmoothingCrossEntropy
from homura.vision.models.cifar_resnet import wrn28_2, wrn28_10, resnet20, resnet56, resnext29_32x4d
from asam import ASAM, SAM
from pyramid_shakedrop import CutMixCrossEntropyLoss, CutMix, ImageNetPolicy, PyramidNet, CIFAR10Policy
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict



def load_cifar(data_loader, batch_size=256, num_workers=2, autoaugment=False):
    if data_loader == CIFAR10:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    # Transforms
    if autoaugment:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                              transforms.RandomHorizontalFlip(),
                                              CIFAR10Policy(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
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
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_set = data_loader(root='/scratch/datasets/CIFAR100/', train=True, download=True, transform=train_transform)
    if args.cutmix:
        train_set = CutMix(train_set, num_class=100, beta=1.0, prob=0.5, num_mix=2)
    test_set = data_loader(root='/scratch/datasets/CIFAR100/', train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    return train_loader, test_loader

def cuda_to_cpu_statedict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def interpolate(model1, model2, model_inter, step):
    assert 0 <= step <= 1
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())
    params_inter = list(model_inter.parameters())
    # normal parameters
    for i, param in enumerate(params_inter):
        assert (param.shape == params2[i].shape) and (param.shape == params1[i].shape)
        param.data = params1[i].data * (1 - step) + (step) * params2[i].data

    # batch norm running mean and var
    def flatten(model):
        flattened = [flatten(children) for children in model.children()]
        res = [model]
        for module in flattened:
            res += module
        return res

    modules1 = flatten(model1)
    modules2 = flatten(model2)
    modules_inter = flatten(model_inter)
    for i, module in enumerate(modules_inter):
        if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
            module.running_mean = modules1[i].running_mean * (1 - step) + modules2[i].running_mean * step
            module.running_var = modules1[i].running_var * (1 - step) + modules2[i].running_var * step

def train(args):

    assert args.m <= args.batch_size
    assert args.batch_size % args.m == 0
    m_factor = args.batch_size / args.m
    if m_factor != 1:
        print(
            'Using {}-sharpness with factor {}, since batch_size={} and m={}!'.format(args.m, m_factor, args.batch_size,
                                                                                      args.m))
    # Data Loader
    train_loader, test_loader = load_cifar(eval(args.dataset), args.m, autoaugment=args.autoaugment)
    num_classes = 10 if args.dataset == 'CIFAR10' else 100

    # Model
    if args.model == 'pyramid':
        # model = PyramidNet('cifar100', 272, 200, 100, True).cuda()
        model1 = PyramidNet('cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
        model2 = PyramidNet('cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
        model_inter = PyramidNet('cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
        model_init = PyramidNet('cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
    else:
        model1 = eval(args.model)(num_classes=num_classes).cuda()
        model2 = eval(args.model)(num_classes=num_classes).cuda()
        model_inter= eval(args.model)(num_classes=num_classes).cuda()
        model_init= eval(args.model)(num_classes=num_classes).cuda()

    # load SAM
    model_name = '{}_{}_{}_lr{}_bsize{}_epochs{}_rho{}_layerwise{}_filterwise{}_elementwise{}_autoaugment{}_cutmix{}_m{}_seed{}'.format(
        args.dataset, args.model, args.minimizer, args.lr, args.batch_size, args.epochs,
        args.rho, args.layerwise, args.filterwise, args.elementwise, args.autoaugment,
        args.cutmix,
        args.m, args.seed)
    model_to_load = os.path.join(args.save, model_name + '.pt')
    state_dict_init = torch.load((model_to_load), map_location=torch.device('cpu'))
    model1.load_state_dict(cuda_to_cpu_statedict(state_dict_init))

    # load NORMAL
    model_to_load = os.path.join(args.save,
                                    '{}_{}_{}_lr{}_bsize{}_epochs{}_rho{}_layerwise{}_filterwise{}_elementwise{}_autoaugment{}_cutmix{}_m{}_seed{}'.format(
                                        args.dataset, args.model, 'NORMAL', args.lr, args.batch_size, int(args.epochs*2),
                                        0.5, False, False, False, args.autoaugment,
                                        args.cutmix,
                                        args.m,args.seed) + '.pt')
    state_dict_init = torch.load(model_to_load, map_location=torch.device('cpu'))
    model2.load_state_dict(cuda_to_cpu_statedict(state_dict_init))

    model_to_load = os.path.join(args.save, '{}_{}_seed{}_init'.format(args.dataset, args.model, args.seed) + '.pt')
    state_dict_init = torch.load(model_to_load, map_location=torch.device('cpu'))
    model_init.load_state_dict((state_dict_init))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model1.to(device)
    model2.to(device)
    model_inter.to(device)

    # Loss Functions
    if args.cutmix:
        print('Using CutMixCrossEntropyLoss')
        criterion = CutMixCrossEntropyLoss(True)
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    with open(os.path.join(args.save, model_name + '_interpolation'+('_frominitSAM' if args.from_init_sam else '') + ('_frominitNormal' if args.from_init_normal else '') +'.csv'), 'w') as f:
        f.write('step,train_loss,test_loss, train_accuracy, test_accuracy\n')

    train_losses=[]
    train_accuracies=[]
    test_losses = []
    test_accuracies = []
    best_accuracy = 0.
    with torch.no_grad():
        for step in np.linspace(0,1,num=args.interpolation_steps):

            # interpolate between models
            interpolate(model_init if args.from_init_normal else model1,model_init if args.from_init_sam else model2,model_inter,step)

            # Train set
            model_inter.eval()
            loss = 0.
            accuracy = 0.
            cnt = 0.
            for idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.type(torch.int64).to(device)
                predictions = model_inter(inputs)
                batch_loss = criterion(predictions, targets)
                loss += batch_loss.sum().item()
                if args.cutmix:
                    accuracy += (torch.argmax(predictions, 1) == torch.argmax(targets, 1)).sum().item()
                else:
                    accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)
            loss /= cnt
            accuracy *= 100. / cnt
            train_losses.append(loss)
            train_accuracies.append(accuracy)
            print(f"Step: {step}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")

            # Test
            model_inter.eval()
            loss = 0.
            accuracy = 0.
            cnt = 0.
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.type(torch.int64).to(device)
                predictions = model_inter(inputs)
                loss += criterion(predictions, targets).sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)
            loss /= cnt
            accuracy *= 100. / cnt
            test_losses.append(loss)
            test_accuracies.append(accuracy)
            print(f"Step: {step}, Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")
            with open(os.path.join(args.save,
                                   model_name + '_interpolation' + ('_frominitSAM' if args.from_init_sam else '') + (
                                   '_frominitNormal' if args.from_init_normal else '') + '.csv'), 'a') as f:
                f.write('{},{},{}, {}, {}\n'.format(step, train_losses[-1], test_losses[-1], train_accuracies[-1], test_accuracies[-1]))
            
    print(f"Best test accuracy: {best_accuracy}")
    plt.figure()
    plt.plot(range(len(train_losses)),train_losses,c='red',label='Train Loss')
    plt.scatter(range(len(train_losses)),train_losses,c='red',label='Train Loss')
    plt.plot(range(len(test_losses)),test_losses,c='blue',label='Test Loss')
    plt.plot(range(len(test_losses)),test_losses,c='blue',label='Test Loss')
    plt.legend()
    plt.savefig(os.path.join(args.save,model_name + 'losses_interpolated'+('_frominitSAM' if args.from_init_sam else '') + (
                                   '_frominitNormal' if args.from_init_normal else '')+'.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='CIFAR10', type=str, help="CIFAR10 or CIFAR100.")
    parser.add_argument("--model", default='wrn28_10', type=str, help="Name of model architecure")
    parser.add_argument("--minimizer", default='ASAM', type=str, help="NORMAL, ASAM or SAM.")
    parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate.")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum.")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay factor.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing.")
    parser.add_argument("--rho", default=0.5, type=float, help="Rho for ASAM.")
    parser.add_argument("--layerwise", action='store_true', help="layerwise normalization for ASAM.")
    parser.add_argument("--filterwise", action='store_true', help="filterwise normalization for ASAM.")
    parser.add_argument("--elementwise", action='store_true', help="elementwise normalization for ASAM.")
    parser.add_argument("--autoaugment", action='store_true', help="apply autoaugment transformation.")
    parser.add_argument("--cutmix", action='store_true', help="apply cutmix transformation.")
    parser.add_argument("--eta", default=0.0, type=float, help="Eta for ASAM.")
    parser.add_argument("--m", default=128, type=int,
                        help="m-sharpness value: Ascent step is averaged over m chunks, each of size batch_size/m")
    parser.add_argument('--save', default='./snapshots/basins', type=str, help='directory to save models in')
    parser.add_argument("--seed", default=123456, type=int, help="Seed")
    parser.add_argument("--interpolation_steps", default=200, type=int,
                        help="number of steps to evaluate between models")
    parser.add_argument("--from_init_sam", action='store_true')
    parser.add_argument("--from_init_normal", action='store_true')

    args = parser.parse_args()
    print(args.elementwise, args.filterwise, args.layerwise)
    print('cutmix: ',args.cutmix)
    assert args.dataset in ['CIFAR10', 'CIFAR100'], \
        f"Invalid data type. Please select CIFAR10 or CIFAR100"
    assert args.minimizer in ['ASAM', 'SAM', 'NORMAL'], \
        f"Invalid minimizer type. Please select NORMAL, ASAM or SAM"

    # Make save directory
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    train(args)
