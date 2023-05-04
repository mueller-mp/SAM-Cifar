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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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
        model = PyramidNet('cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
    else:
        model = eval(args.model)(num_classes=num_classes).cuda()

    # initial model
    if args.save_init:
        torch.save(model.state_dict(),
                   os.path.join(args.save, '{}_{}_seed{}_init'.format(args.dataset, args.model,args.seed) + '.pt'))

    if args.load_init:
        model_to_load = os.path.join(args.save, '{}_{}_seed{}_init'.format(args.dataset, args.model,args.seed) + '.pt')
        state_dict_init = torch.load(model_to_load, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict_init)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(model)

    model.to(device)

    # Minimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    if args.minimizer != 'NORMAL':
        minimizer = eval(args.minimizer)(optimizer, model, rho=args.rho, eta=args.eta, layerwise=args.layerwise,
                                     elementwise=args.elementwise, filterwise=args.filterwise, adaptive=True if args.minimizer=='ASAM' else False)
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Loss Functions
    if args.cutmix:
        print('Using CutMixCrossEntropyLoss')
        criterion = CutMixCrossEntropyLoss(True)
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    losses=[]
    adv_losses=[]
    best_accuracy = 0.
    for epoch in range(args.epochs):
        # Train
        model.train()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.type(torch.int64).to(device)
            # Ascent Step
            predictions = model(inputs)
            batch_loss = criterion(predictions, targets)
            batch_loss.mean().backward()
            if args.minimizer in ['SAM', 'ASAM']:
                minimizer.ascent_step()
                losses.append(batch_loss.mean().item())
                # Descent Step
                batch_loss_descent = criterion(model(inputs), targets).mean()
                batch_loss_descent.backward()
                adv_losses.append(batch_loss_descent.mean().item())
                if m_factor == 1:
                    minimizer.descent_step()
                elif m_factor > 1:
                    minimizer.accumulate_grad_and_resume()
                    if (idx + 1) % m_factor == 0:  #
                        minimizer.descent_with_accumulated_grad(steps=m_factor)
                else:
                    raise ValueError('m has to be an integer <= batch size, but is m={}'.format(args.m))
            else:
                optimizer.step()
                optimizer.zero_grad()
            with torch.no_grad():
                loss += batch_loss.sum().item()
                if args.cutmix:
                    accuracy += (torch.argmax(predictions, 1) == torch.argmax(targets, 1)).sum().item()
                else:
                    accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
        loss /= cnt
        accuracy *= 100. / cnt
        print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
        scheduler.step()

        # Test
        model.eval()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.type(torch.int64).to(device)
                predictions = model(inputs)
                loss += criterion(predictions, targets).sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)
            loss /= cnt
            accuracy *= 100. / cnt
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            # Save current model
            torch.save(model.state_dict(),
                       os.path.join(args.save,
                                    '{}_{}_{}_lr{}_bsize{}_epochs{}_rho{}_layerwise{}_filterwise{}_elementwise{}_autoaugment{}_cutmix{}_m{}_seed{}'.format(
                                        args.dataset, args.model, args.minimizer, args.lr, args.batch_size, args.epochs,
                                        args.rho, args.layerwise, args.filterwise, args.elementwise, args.autoaugment,
                                        args.cutmix,
                                        args.m,args.seed) + '.pt'))
        print(f"Epoch: {epoch}, Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")
    print(f"Best test accuracy: {best_accuracy}")
    plt.figure()
    plt.plot(range(len(losses)),losses,c='red',label='Loss')
    plt.scatter(range(len(losses)),losses,c='red',label='Loss')
    plt.plot(range(len(adv_losses)),adv_losses,c='blue',label='Ascent Loss')
    plt.plot(range(len(adv_losses)),adv_losses,c='blue',label='Ascent Loss')
    plt.savefig(os.path.join(args.save,
                                    '{}_{}_{}_lr{}_bsize{}_epochs{}_rho{}_layerwise{}_filterwise{}_elementwise{}_autoaugment{}_cutmix{}_m{}_seed{}_'.format(
                                        args.dataset, args.model, args.minimizer, args.lr, args.batch_size, args.epochs,
                                        args.rho, args.layerwise, args.filterwise, args.elementwise, args.autoaugment,
                                        args.cutmix,
                                        args.m,args.seed) + 'losses.png'))


def train_all(args):
    assert args.m <= args.batch_size
    assert args.batch_size % args.m == 0

    prefix = datetime.now().strftime("%y-%m-%d_%H:%M:%S/")
    base_folder = 'runs/' + prefix
    writer = SummaryWriter(base_folder)

    m_factor = args.batch_size / args.m
    if m_factor != 1:
        print(
            'Using {}-sharpness with factor {}, since batch_size={} and m={}!'.format(args.m, m_factor, args.batch_size,
                                                                                      args.m))
    # Data Loader
    train_loader, test_loader = load_cifar(eval(args.dataset), args.m, autoaugment=args.autoaugment)
    num_classes = 10 if args.dataset == 'CIFAR10' else 100

    # Model - initialize 3 models with same parameters
    if args.model == 'pyramid':
        model = PyramidNet('cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
        model_sam = PyramidNet('cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
        model_normal = PyramidNet('cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
        model_normal.load_state_dict(model_sam.state_dict())
        model_finetune = PyramidNet('cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
        model_finetune.load_state_dict(model_sam.state_dict())

    else:
        model_sam = eval(args.model)(num_classes=num_classes).cuda()
        model_normal = eval(args.model)(num_classes=num_classes).cuda()
        model_normal.load_state_dict(model_sam.state_dict())
        model_finetune = eval(args.model)(num_classes=num_classes).cuda()
        model_finetune.load_state_dict(model_sam.state_dict())


    # initial model
    if args.save_init:
        torch.save(model_sam.state_dict(),
                   os.path.join(args.save, '{}_{}_seed{}_init'.format(args.dataset, args.model,args.seed) + '.pt'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model_sam = torch.nn.DataParallel(model_sam)
        model_normal = torch.nn.DataParallel(model_normal)
        model_finetune = torch.nn.DataParallel(model_finetune)

    model_sam.to(device)
    model_normal.to(device)
    model_finetune.to(device)

    # Minimizer
    optimizer_sam = torch.optim.SGD(model_sam.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_normal = torch.optim.SGD(model_normal.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_finetune = torch.optim.SGD(model_finetune.parameters(), lr=0.0001,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    minimizer_sam = eval(args.minimizer)(optimizer_sam, model_sam, rho=args.rho, eta=args.eta, layerwise=args.layerwise,
                                     elementwise=args.elementwise, filterwise=args.filterwise, adaptive=True if args.minimizer=='ASAM' else False)
    minimizer_normal = optimizer_normal
    minimizer_finetune = eval(args.minimizer)(optimizer_finetune, model_finetune, rho=args.rho, eta=args.eta, layerwise=args.layerwise,
                                         elementwise=args.elementwise, filterwise=args.filterwise,
                                         adaptive=True if args.minimizer == 'ASAM' else False)
    # Learning Rate Scheduler
    scheduler_sam = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_sam, args.epochs)
    scheduler_normal = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_normal, args.epochs*2)
    scheduler_finetune = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_finetune, args.ft_epochs)

    # Loss Functions
    if args.cutmix:
        print('Using CutMixCrossEntropyLoss')
        criterion = CutMixCrossEntropyLoss(True)
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    losses_sam=[]
    adv_losses_sam=[]
    adv_losses_finetune=[]
    losses_normal=[]
    losses_finetune=[]
    best_accuracy_sam = 0.
    best_accuracy_normal = 0.
    best_accuracy_finetune = 0.
    for epoch in range(2*args.epochs):
        # Train
        model_sam.train()
        model_normal.train()
        loss_sam = 0.
        accuracy_sam = 0.
        loss_normal = 0.
        accuracy_normal = 0.
        loss_sam_adv = 0.
        cnt = 0.
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.type(torch.int64).to(device)
            if epoch < args.epochs: # sam runs for half the epochs of normal
                # Ascent Step
                predictions = model_sam(inputs)
                batch_loss = criterion(predictions, targets)
                batch_loss.mean().backward()
                minimizer_sam.ascent_step()
                losses_sam.append(batch_loss.mean().item())
                # Descent Step
                batch_loss_descent = criterion(model_sam(inputs), targets)
                batch_loss_descent.mean().backward()
                adv_losses_sam.append(batch_loss_descent.mean().item())
                if m_factor == 1:
                    minimizer_sam.descent_step()
                elif m_factor > 1:
                    minimizer_sam.accumulate_grad_and_resume()
                    if (idx + 1) % m_factor == 0:  #
                        minimizer_sam.descent_with_accumulated_grad(steps=m_factor)
                else:
                    raise ValueError('m has to be an integer <= batch size, but is m={}'.format(args.m))
                with torch.no_grad():
                    loss_sam += batch_loss.sum().item()
                    loss_sam_adv += batch_loss_descent.sum().item()
                    if args.cutmix:
                        accuracy_sam += (torch.argmax(predictions, 1) == torch.argmax(targets, 1)).sum().item()
                    else:
                        accuracy_sam += (torch.argmax(predictions, 1) == targets).sum().item()
            # normal
            predictions = model_normal(inputs)
            batch_loss = criterion(predictions, targets)
            batch_loss.mean().backward()
            losses_normal.append(batch_loss.mean().item())
            minimizer_normal.step()
            minimizer_normal.zero_grad()
            with torch.no_grad():
                loss_normal += batch_loss.sum().item()
                if args.cutmix:
                    accuracy_normal += (torch.argmax(predictions, 1) == torch.argmax(targets, 1)).sum().item()
                else:
                    accuracy_normal += (torch.argmax(predictions, 1) == targets).sum().item()

            # save for finetuning later
            if epoch==args.epochs-args.ft_epochs-1:
                state_dict_for_ft = model_normal.state_dict()
            cnt += len(targets)

        loss_normal /= cnt
        accuracy_normal *= 100. / cnt
        print(f"Epoch: {epoch}, Train acc normal: {accuracy_normal:6.2f} %, Train loss normal: {loss_normal:8.5f}")
        scheduler_normal.step()



        if epoch < args.epochs:
            loss_sam /= cnt
            accuracy_sam *= 100. / cnt
            loss_sam_adv /= cnt
            print(f"Epoch: {epoch}, Train acc sam: {accuracy_sam:6.2f} %, Train loss sam: {loss_sam:8.5f}, Train loss adv. sam {loss_sam_adv:8.5f}")
            scheduler_sam.step()

        writer.add_scalar('train/loss_normal', loss_normal, global_step=epoch)
        writer.add_scalar('train/acc_normal', accuracy_normal, global_step=epoch)
        writer.add_scalar('train/loss_sam', loss_sam, global_step=epoch)
        writer.add_scalar('train/acc_sam', accuracy_sam, global_step=epoch)

        # Test
        model_sam.eval()
        model_normal.eval()
        loss_sam = 0.
        accuracy_sam = 0.
        loss_normal = 0.
        accuracy_normal = 0.
        cnt = 0.
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.type(torch.int64).to(device)
                #sam
                predictions = model_sam(inputs)
                loss_sam += criterion(predictions, targets).sum().item()
                accuracy_sam += (torch.argmax(predictions, 1) == targets).sum().item()
                # normal
                predictions = model_normal(inputs)
                loss_normal += criterion(predictions, targets).sum().item()
                accuracy_normal += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)

            loss_sam /= cnt
            accuracy_sam *= 100. / cnt
            loss_normal /= cnt
            accuracy_normal *= 100. / cnt

            writer.add_scalar('test/loss_normal', loss_normal, global_step=epoch)
            writer.add_scalar('test/acc_normal', accuracy_normal, global_step=epoch)
            writer.add_scalar('test/loss_sam', loss_sam, global_step=epoch)
            writer.add_scalar('test/acc_sam', accuracy_sam, global_step=epoch)

        if best_accuracy_sam < accuracy_sam:
            best_accuracy_sam = accuracy_sam
            # Save current model
            torch.save(model_sam.state_dict(),
                       os.path.join(args.save,
                                    '{}_{}_{}_lr{}_bsize{}_epochs{}_rho{}_layerwise{}_filterwise{}_elementwise{}_autoaugment{}_cutmix{}_m{}_seed{}'.format(
                                        args.dataset, args.model, args.minimizer, args.lr, args.batch_size, args.epochs,
                                        args.rho, args.layerwise, args.filterwise, args.elementwise, args.autoaugment,
                                        args.cutmix,
                                        args.m,args.seed) + '.pt'))
        print(f"SAM, Epoch: {epoch}, Test accuracy:  {accuracy_sam:6.2f} %, Test loss:  {loss_sam:8.5f}")
        if best_accuracy_normal < accuracy_normal:
            best_accuracy_normal = accuracy_normal
            # Save current model
            torch.save(model_normal.state_dict(),
                       os.path.join(args.save,
                                    '{}_{}_{}_lr{}_bsize{}_epochs{}_rho{}_layerwise{}_filterwise{}_elementwise{}_autoaugment{}_cutmix{}_m{}_seed{}'.format(
                                        args.dataset, args.model, 'NORMAL', args.lr, args.batch_size, args.epochs*2,
                                        args.rho, args.layerwise, args.filterwise, args.elementwise, args.autoaugment,
                                        args.cutmix,
                                        args.m, args.seed) + '.pt'))
        print(f"NORMAL, Epoch: {epoch}, Test accuracy:  {accuracy_normal:6.2f} %, Test loss:  {loss_normal:8.5f}")
    print(f"Best test accuracy sam: {best_accuracy_sam}")
    print(f"Best test accuracy normal: {best_accuracy_normal}")
    # plt.figure()
    # plt.plot(range(len(losses_sam)),losses_sam,c='red',label='Loss')
    # plt.scatter(range(len(losses_sam)),losses_sam,c='red',label='Loss')
    # plt.plot(range(len(adv_losses_sam)),adv_losses_sam,c='blue',label='Ascent Loss')
    # plt.plot(range(len(adv_losses_sam)),adv_losses_sam,c='blue',label='Ascent Loss')
    # plt.savefig(os.path.join(args.save,
    #                                 '{}_{}_{}_lr{}_bsize{}_epochs{}_rho{}_layerwise{}_filterwise{}_elementwise{}_autoaugment{}_cutmix{}_m{}_seed{}_'.format(
    #                                     args.dataset, args.model, args.minimizer, args.lr, args.batch_size, args.epochs,
    #                                     args.rho, args.layerwise, args.filterwise, args.elementwise, args.autoaugment,
    #                                     args.cutmix,
    #                                     args.m,args.seed) + 'losses_sam.png'))
    # plt.close()
    # plt.figure()
    # plt.plot(range(len(losses_normal)),losses_normal,c='red',label='Loss')
    # plt.scatter(range(len(losses_normal)),losses_normal,c='red',label='Loss')
    # plt.savefig(os.path.join(args.save,
    #                                 '{}_{}_{}_lr{}_bsize{}_epochs{}_rho{}_layerwise{}_filterwise{}_elementwise{}_autoaugment{}_cutmix{}_m{}_seed{}_'.format(
    #                                     args.dataset, args.model, 'NORMAL', args.lr, args.batch_size, args.epochs*2,
    #                                     args.rho, args.layerwise, args.filterwise, args.elementwise, args.autoaugment,
    #                                     args.cutmix,
    #                                     args.m,args.seed) + 'losses_normal.png'))
    # plt.close()

    # finetune
    model_finetune.load_state_dict(state_dict_for_ft)
    for epoch in range(args.ft_epochs):
        # Train
        model_finetune.train()
        loss_finetune = 0.
        accuracy_finetune = 0.
        loss_finetune_adv = 0.

        cnt = 0.
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.type(torch.int64).to(device)
            # Ascent Step
            predictions = model_finetune(inputs)
            batch_loss = criterion(predictions, targets)
            batch_loss.mean().backward()
            minimizer_finetune.ascent_step()
            losses_finetune.append(batch_loss.mean().item())
            # Descent Step
            batch_loss_descent = criterion(model_finetune(inputs), targets)
            batch_loss_descent.mean().backward()
            adv_losses_finetune.append(batch_loss_descent.mean().item())
            if m_factor == 1:
                minimizer_finetune.descent_step()
            elif m_factor > 1:
                minimizer_finetune.accumulate_grad_and_resume()
                if (idx + 1) % m_factor == 0:  #
                    minimizer_finetune.descent_with_accumulated_grad(steps=m_factor)
            else:
                raise ValueError('m has to be an integer <= batch size, but is m={}'.format(args.m))
            with torch.no_grad():
                loss_finetune += batch_loss.sum().item()
                loss_finetune_adv += batch_loss_descent.sum().item()
                if args.cutmix:
                    accuracy_finetune += (torch.argmax(predictions, 1) == torch.argmax(targets, 1)).sum().item()
                else:
                    accuracy_finetune += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)

        loss_finetune /= cnt
        loss_finetune_adv /= cnt
        accuracy_finetune *= 100. / cnt
        print(f"FINTETUNE, Epoch: {epoch}, Train acc: {accuracy_finetune:6.2f} %, Train loss: {loss_finetune:8.5f}, Train loss finetune adv {loss_finetune_adv:8.5f}")
        scheduler_finetune.step()
        writer.add_scalar('train/loss_ft', loss_finetune, global_step=epoch+args.epochs-args.ft_epochs)
        writer.add_scalar('train/acc_ft', accuracy_finetune, global_step=epoch+args.epochs-args.ft_epochs)

        # Test
        model_finetune.eval()
        loss_finetune = 0.
        accuracy_finetune = 0.
        cnt = 0.
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.type(torch.int64).to(device)
                predictions = model_finetune(inputs)
                loss_finetune += criterion(predictions, targets).sum().item()
                accuracy_finetune += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)

            loss_finetune /= cnt
            accuracy_finetune *= 100. / cnt

            writer.add_scalar('test/loss_ft', loss_finetune, global_step=epoch + args.epochs - args.ft_epochs)
            writer.add_scalar('test/acc_ft', accuracy_finetune, global_step=epoch + args.epochs - args.ft_epochs)

        if best_accuracy_finetune < accuracy_finetune:
            best_accuracy_finetune = accuracy_finetune
            # Save current model
            torch.save(model_finetune.state_dict(),
                       os.path.join(args.save,
                                    '{}_{}_{}_lr{}_ftepochs_{}_bsize{}_epochs{}_rho{}_layerwise{}_filterwise{}_elementwise{}_autoaugment{}_cutmix{}_m{}_seed{}'.format(
                                        args.dataset, args.model, 'FINETUNE', args.lr, args.ft_epochs, args.batch_size, args.epochs,
                                        args.rho, args.layerwise, args.filterwise, args.elementwise, args.autoaugment,
                                        args.cutmix,
                                        args.m, args.seed) + '.pt'))
        print(f"FINETUNING, Epoch: {epoch}, Test accuracy:  {accuracy_finetune:6.2f} %, Test loss:  {loss_finetune:8.5f}")

    print(f"Best test accuracy: {best_accuracy_finetune}")
    # plt.figure()
    # plt.plot(range(len(losses_finetune)), losses_finetune, c='red', label='Loss')
    # plt.scatter(range(len(losses_finetune)), losses_finetune, c='red', label='Loss')
    # plt.plot(range(len(adv_losses_finetune)), adv_losses_finetune, c='blue', label='Ascent Loss')
    # plt.plot(range(len(adv_losses_finetune)), adv_losses_finetune, c='blue', label='Ascent Loss')
    # plt.savefig(os.path.join(args.save,
    #                                 '{}_{}_{}_lr{}_ftepochs_{}_bsize{}_epochs{}_rho{}_layerwise{}_filterwise{}_elementwise{}_autoaugment{}_cutmix{}_m{}_seed{}'.format(
    #                                     args.dataset, args.model, 'FINETUNE', args.lr, args.ft_epochs, args.batch_size, args.epochs,
    #                                     args.rho, args.layerwise, args.filterwise, args.elementwise, args.autoaugment,
    #                                     args.cutmix,
    #                                     args.m, args.seed) + 'losses_ft.png'))

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
    parser.add_argument('--save', default='./snapshots', type=str, help='directory to save models in')
    parser.add_argument("--seed", default=123456, type=int, help="Seed")
    parser.add_argument("--save_init", action='store_true', help="save initialization")
    parser.add_argument("--load_init", action='store_true', help="load initialization")
    parser.add_argument("--ft_epochs", default=20, type=int, help="Number of fine-tuning epochs.")

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
    # train(args)
    train_all(args)
