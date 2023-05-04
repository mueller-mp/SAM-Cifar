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

def load_cifar(data_loader, batch_size=256, num_workers=2, autoaugment=False, data_path = '/scratch/datasets/CIFAR100/'):
    if data_loader == CIFAR10:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    # Transforms
    if autoaugment:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=0),
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
    train_set = data_loader(root=data_path, train=True, download=False, transform=train_transform)
    if args.cutmix:
        train_set = CutMix(train_set, num_class=100, beta=1.0, prob=0.5, num_mix=2)
    test_set = data_loader(root=data_path, train=False, download=False, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader


def train(args):
    state = {k: v for k, v in args._get_kwargs()}
    print(state)
    assert args.m <= args.batch_size
    assert args.batch_size % args.m == 0
    n_chunks = args.batch_size / args.m # number of chunks over which m-sharpness is computed
    if n_chunks != 1:
        print(
            'Using {}-sharpness with factor {}, since batch_size={} and m={}!'.format(args.m, n_chunks, args.batch_size,
                                                                                      args.m))
    model_name = '{}_{}_{}_lr{}_bsize{}_epochs{}_rho{}_layerwise{}_filterwise{}_elementwise{}_autoaugment{}_cutmix{}_m{}_eta{}_{}_seed{}'.format(
                                        args.dataset, args.model, args.minimizer, args.lr, args.batch_size, args.epochs,
                                        args.rho, args.layerwise, args.filterwise, args.elementwise, args.autoaugment,
                                        args.cutmix,
                                        args.m, args.eta, args.norm_adaptive, args.seed)
    prefix = datetime.now().strftime("%y-%m-%d_%H:%M:%S/")+model_name
    base_folder = args.save+'/runs/' + prefix
    writer = SummaryWriter(base_folder)
    # Data Loader
    train_loader, test_loader = load_cifar(eval(args.dataset), args.m, autoaugment=args.autoaugment, data_path=args.data_path)
    num_classes = 10 if args.dataset == 'CIFAR10' else 100

    # Model
    if args.model == 'pyramid':
        model = PyramidNet('cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
    else:
        model = eval(args.model)(num_classes=num_classes).cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(model)

    model.to(device)

    # Minimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    if args.minimizer == 'SGD':
        minimizer = optimizer
    else:
        minimizer = eval(args.minimizer)(optimizer, model, rho=args.rho, eta=args.eta, layerwise=args.layerwise,
                                     elementwise=args.elementwise, filterwise=args.filterwise, p=args.p, normalize_bias=args.normalize_bias)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer if args.minimizer=='SGD' else minimizer.optimizer, args.epochs)

    norm_adaptive = eval('attacks.weights.norms.'+args.norm_adaptive)()

    # Loss Functions
    if args.cutmix:
        print('Using CutMixCrossEntropyLoss')
        criterion = CutMixCrossEntropyLoss(True)
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    best_accuracy = 0.
    loss_best = 0.
    for epoch in range(args.epochs):
        # Train
        model.train()
        loss = 0.
        loss_adv = 0.
        accuracy = 0.
        cnt = 0.
        for idx, (inputs, targets) in enumerate(train_loader):
            unperturbed_model = deepcopy(model)
            inputs = inputs.to(device)
            targets = targets.type(torch.int64).to(device)
            # Ascent Step
            if args.eval_ascent:
                model.eval()
            predictions = model(inputs)
            batch_loss = criterion(predictions, targets)
            batch_loss.mean().backward()
            if args.minimizer=='SGD':
                minimizer.step()
                minimizer.zero_grad()
            else:
                minimizer.ascent_step()
            model.train()
            norm = norm_adaptive(unperturbed_model, model, layers = [i for i in range(len(list(model.parameters())))]).item()
            if args.minimizer !='SGD':
                # Descent Step
                batch_loss_2 = criterion(model(inputs), targets)
                batch_loss_2.mean().backward()
                if n_chunks == 1:
                    minimizer.descent_step()
                elif n_chunks > 1:
                    minimizer.accumulate_grad_and_resume()
                    if (idx + 1) % n_chunks == 0:  #
                        minimizer.descent_with_accumulated_grad(steps=n_chunks)
                else:
                    raise ValueError('m has to be an integer <= batch size, but is m={}'.format(args.m))
            with torch.no_grad():
                loss += batch_loss.sum().item()
                if args.minimizer == 'SGD':
                    loss_adv=0
                    batch_loss_2=torch.tensor(0.)
                else:
                    loss_adv += batch_loss_2.sum().item()
                if args.cutmix:
                    accuracy += (torch.argmax(predictions, 1) == torch.argmax(targets, 1)).sum().item()
                else:
                    accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            writer.add_scalar('train/Batch_loss', batch_loss.mean().item(), global_step=epoch*len(train_loader)+idx)
            writer.add_scalar('train/Batch_loss_adversarial', batch_loss_2.mean().item(), global_step=(epoch)*len(train_loader)+idx)
            writer.add_scalar('train/norm_adaptive', norm, global_step=epoch*len(train_loader)+idx)
            print('Norm adaptive: ', norm)
            cnt += len(targets)
        if args.smoothing: # smoothing loss does reduce implicitly
            loss /= (idx+1)
            loss_adv /= (idx+1)
        else:
            loss /= cnt
            loss_adv /=cnt
        accuracy *= 100. / cnt
        print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}, Train loss adv. {loss_adv:8.5f}")
        scheduler.step()
        state['loss_train_adv_batch'] = batch_loss_2.mean().item()
        state['loss_train_adv_epoch'] = loss_adv
        state['loss_train'] = loss
        state['loss_train_batch'] = batch_loss.mean().item()
        state['accuracy_train'] = accuracy
        writer.add_scalar('train/loss_epoch', loss, global_step=epoch)
        writer.add_scalar('train/loss_epoch_adv', loss_adv, epoch)
        writer.add_scalar('train/accuracy', accuracy, global_step=epoch)
        # Test
        model.eval()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.type(torch.int64).to(device)
                predictions = model(inputs)
                loss += criterion(predictions, targets).sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)
            if args.smoothing:  # smoothing loss does reduce implicitly
                loss /= (idx + 1)
            else:
                loss /= cnt
            accuracy *= 100. / cnt
        writer.add_scalar('test/loss_epoch', loss, global_step=epoch)
        writer.add_scalar('test/accuracy', accuracy, global_step=epoch)
        state['accuracy_test']=accuracy
        state['loss_test']=loss
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            # Save current best model
            torch.save(model.state_dict(),
                       os.path.join(args.save,
                                    model_name + '_best.pt'))
            loss_best = loss

        state['accuracy_test_best']=best_accuracy
        state['loss_test_best']=loss_best
        print(f"Epoch: {epoch}, Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")

    # Save last model
    torch.save(model.state_dict(),
               os.path.join(args.save,
                            model_name + '_last.pt'))

    # save final state
    filename = args.save + '/summary.csv'
    file_exists = os.path.isfile(filename)

    with open(filename, 'a') as f:
        if not file_exists:
            # write header
            f.write(','.join([str(i) for i in list(state.keys())]) + '\n')
        f.write(','.join([str(i) for i in list(state.values())]) + '\n')

    print(f"Best test accuracy: {best_accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='CIFAR10', type=str, help="CIFAR10 or CIFAR100.")
    parser.add_argument("--data_path", default='/scratch/datasets/CIFAR100/', type=str, help="path to data root.")
    parser.add_argument("--model", default='wrn28_10', type=str, help="Name of model architecure")
    parser.add_argument("--minimizer", default='ASAM', type=str, help="ASAM, SAM or SGD.")
    parser.add_argument("--p", default='2', type=str, choices=['2', 'infinity'])
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
    parser.add_argument("--normalize_bias", action='store_true', help="apply ASAM also to bias params")
    parser.add_argument("--eta", default=0.0, type=float, help="Eta for ASAM.")
    parser.add_argument("--m", default=128, type=int,
                        help="m-sharpness value: Ascent step is averaged over m chunks, each of size batch_size/m")
    parser.add_argument('--save', default='./snapshots', type=str, help='directory to save models in')
    parser.add_argument("--norm_adaptive", default='ElementWiseL2NormAsam', type=str, help="ElementWiseL2NormAsam, LayerWiseL2NormAsam, L2Norm, FilterwiseLinfNormAsam")
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument("--eval_ascent", action='store_true', help="perform ascent step in eval mode")
    args = parser.parse_args()
    print(args.elementwise, args.filterwise, args.layerwise)
    print('cutmix: ', args.cutmix)
    assert args.dataset in ['CIFAR10', 'CIFAR100'], \
        f"Invalid data type. Please select CIFAR10 or CIFAR100"
    assert args.minimizer in ['ASAM', 'SAM', 'SGD'], \
        f"Invalid minimizer type. Please select ASAM or SAM"

    # set seed
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)

    # Make save directory
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)
    train(args)
