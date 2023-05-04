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
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','robust-generalization-flatness'))
import attacks.weights
from copy import deepcopy
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist


class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default
    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.
    Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    Example::
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


from torch.utils.data.distributed import DistributedSampler
def prepare(rank, world_size, batch_size=32, pin_memory=False, num_workers=0, autoaugment=False, cutmix = False, data_path = '/scratch/datasets/CIFAR100/'):
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
    train_set = CIFAR100(root=data_path, train=True, download=False, transform=train_transform)
    if cutmix:
        train_set = CutMix(train_set, num_class=100, beta=1.0, prob=0.5, num_mix=2)
    test_set = CIFAR100(root=data_path, train=False, download=False, transform=test_transform)

    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=0, drop_last=False)
    dataloader = DataLoader(train_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, sampler=sampler)
    test_sampler = DistributedEvalSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader_test = DataLoader(test_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False,
                        shuffle=False, sampler=test_sampler)
    return dataloader, dataloader_test

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args_parser):
    # setup the process groups
    setup(rank, world_size)  # prepare the dataloader
    train_loader, test_loader = prepare(rank, world_size, batch_size=128, autoaugment=args_parser.autoaugment, cutmix=args_parser.cutmix)

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    if args_parser.model == 'pyramid':
        model = PyramidNet('cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True).to(rank)
    else:
        # model = eval(args_parser.model)(num_classes=100).cuda()
        model = wrn28_10(num_classes=100).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    optimizer = torch.optim.SGD(model.parameters(), args_parser.lr,
                                momentum=args_parser.momentum,
                                weight_decay=args_parser.weight_decay)
    minimizer = ASAM(optimizer, model, rho=args_parser.rho, eta=args_parser.eta, layerwise=args_parser.layerwise,
                                     elementwise=args_parser.elementwise, filterwise=args_parser.filterwise, p=args_parser.p,
                                     normalize_bias=args_parser.normalize_bias)

    if args_parser.cutmix:
        print('Using CutMixCrossEntropyLoss')
        criterion = CutMixCrossEntropyLoss(True)
    elif args_parser.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args_parser.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, args_parser.epochs)


    for epoch in range(args_parser.epochs):

        # if we are using DistributedSampler, we have to tell it which epoch this is
        train_loader.sampler.set_epoch(epoch)
        #train
        model.train()
        for step, (x, label) in enumerate(train_loader):
            x = x.to(rank)
            label = label.to(rank)
            # optimizer.zero_grad(set_to_none=True)
            #
            # pred = model(x)
            #
            # loss = criterion(pred, label)
            # loss.mean().backward()
            # optimizer.step()

            # SAM
            predictions = model(x)
            batch_loss = criterion(predictions, label)
            with model.no_sync():
                batch_loss.mean().backward()
            minimizer.ascent_step()

            batch_loss_2 = criterion(model(x), label)
            batch_loss_2.mean().backward()
            minimizer.descent_step()

        # Test
        model.eval()
        loss = 0.
        accuracy_ = 0.
        cnt = 0.
        counter = torch.zeros((2,), device = torch.device(f'cuda:{rank}')) # for reduce
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(rank)
                targets = targets.type(torch.int64).to(rank)
                predictions = model(inputs)
                # loss += criterion(predictions, targets).sum().item()
                accuracy_ += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)
                counter[0] += accuracy_
                counter[1] += cnt
            # if args_parser.smoothing:  # smoothing loss does reduce implicitly
            #     loss /= (idx + 1)
            # else:
            #     loss /= cnt
            accuracy = accuracy_* 100. / cnt
        scheduler.step()

        print('finished epoch ',epoch, ', accuracy = ', accuracy)
        torch.distributed.reduce(counter, 0)
        if rank==0:
            print(f'total accuracy = {100*counter[0]/counter[1]}')
    cleanup()

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
    parser.add_argument("--rho", default=0.2, type=float, help="Rho for ASAM.")
    parser.add_argument("--layerwise", action='store_true', help="layerwise normalization for ASAM.")
    parser.add_argument("--filterwise", action='store_true', help="filterwise normalization for ASAM.")
    parser.add_argument("--elementwise", action='store_true', help="elementwise normalization for ASAM.")
    parser.add_argument("--autoaugment", action='store_true', help="apply autoaugment transformation.")
    parser.add_argument("--cutmix", action='store_true', help="apply cutmix transformation.")
    parser.add_argument("--normalize_bias", action='store_true', help="apply ASAM also to bias params")
    parser.add_argument("--eta", default=0.0, type=float, help="Eta for ASAM.")
    parser.add_argument("--ngpus", default=8, type=int,
                        help="m-sharpness value: Ascent step is averaged over m chunks, each of size batch_size/m")
    parser.add_argument('--save', default='./snapshots', type=str, help='directory to save models in')
    parser.add_argument("--norm_adaptive", default='ElementWiseL2NormAsam', type=str, help="ElementWiseL2NormAsam, LayerWiseL2NormAsam, L2Norm, FilterwiseLinfNormAsam")
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument("--eval_ascent", action='store_true', help="perform ascent step in eval mode")
    args_parser = parser.parse_args()
    print(args_parser.elementwise, args_parser.filterwise, args_parser.layerwise)
    print('cutmix: ', args_parser.cutmix)
    assert args_parser.dataset in ['CIFAR10', 'CIFAR100'], \
        f"Invalid data type. Please select CIFAR10 or CIFAR100"
    assert args_parser.minimizer in ['ASAM', 'SAM', 'SGD'], \
        f"Invalid minimizer type. Please select ASAM or SAM"

    # set seed
    torch.manual_seed(args_parser.seed)
    numpy.random.seed(args_parser.seed)

    # Make save directory
    if not os.path.exists(args_parser.save):
        os.makedirs(args_parser.save)
    if not os.path.isdir(args_parser.save):
        raise Exception('%s is not a dir' % args_parser.save)
    # train(args)
    # suppose we have 3 gpus
    world_size = args_parser.ngpus

    start = time.time()
    mp.spawn(
        main,
        args=(world_size,args_parser),
        nprocs=world_size,
        join=True
    )

    end = time.time()
    print('Time elapsed with world_size ', world_size, ' is ', end - start)