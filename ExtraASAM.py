import argparse
import numpy.random
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from timm.loss import LabelSmoothingCrossEntropy
from homura.vision.models.cifar_resnet import wrn28_2, wrn28_10#, resnet20, resnet56, resnext29_32x4d, cifar_resnet50, resnext29_8x64d
from homura_resnet import resnet56_nosequential, resnet110_nosequential, resnet20_nosequential, resnet32_nosequential, resnext29_32x4d_nosequential
from homura_densenet import densenet100_nosequential, densenet40_nosequential
from asam_pyramid import ASAM, SAM, ExtraASAM, ExtraSAM, SAM_BN, ASAM_BN, AVG_ASAM_BN, AVG_SAM_BN, SAM_BN_WEIGHTS, ASAM_BN_WEIGHTS, ASAM_BN_FC, SAM_BN_FC, FISHER_SAM
from pyramid_shakedrop import CutMixCrossEntropyLoss, CutMix, ImageNetPolicy, PyramidNet, CIFAR10Policy
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','robust-generalization-flatness'))
import attacks.weights
from copy import deepcopy
from models_GN import wrn28_10_GN
from models_trades import PreActResNet34BatchNorm, PreActResNet34GroupNorm, PreActResNet34LayerNorm, PreActResNet34LayerNormSmall, init_weights
from models_convmixer import ConvMixerBatch, ConvMixerGroup, ConvMixerLayer
def load_cifar(data_loader, batch_size=256, num_workers=2, autoaugment=False, data_path = '/scratch/datasets/CIFAR100/', random_idxs_frac = 0.):
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
    if random_idxs_frac > 0.: # randomize the last frac of train idxs
        num_train_samples = len(train_set.targets)
        num_random = int(numpy.floor(random_idxs_frac * num_train_samples))
        random_idxs = list(numpy.random.randint(0, 99, size=[num_random]))
        new_targets = train_set.targets[:(num_train_samples - num_random)] + random_idxs
        train_set.targets = new_targets
    if args.cutmix:
        assert random_idxs_frac<=0.
        train_set = CutMix(train_set, num_class=100, beta=1.0, prob=0.5, num_mix=2)
    test_set = data_loader(root=data_path, train=False, download=False, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader

def angle_between_grads(grads_1, grads_2):
    # model has current grads, minimizer old grads
    norm1=[]
    norm2=[]
    inner_prod = 0.
    for g1, g2 in zip(grads_1, grads_2):
        assert g1.shape==g2.shape
        inner_prod += torch.sum(g1*g2)
        norm1.append(g1.norm(p=2))
        norm2.append(g2.norm(p=2))

    norm1 = torch.norm(torch.stack(norm1),p=2)
    norm2 = torch.norm(torch.stack(norm2),p=2)
    assert norm1>0 and norm2>0
    # print(norm1)
    # print(norm2)
    cosine = inner_prod / (norm1 * norm2)
    return cosine

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
    # model_name = '{}_{}_{}_p{}_lr{}_bsize{}_epochs{}_rho{}_layerwise{}_filterwise{}_elementwise{}_autoaugment{}_cutmix{}_m{}_eta{}_{}_seed{}{}{}{}{}'.format(
    #                                     args.dataset, args.model, args.minimizer, args.p, args.lr, args.batch_size, args.epochs,
    #                                     args.rho, args.layerwise, args.filterwise, args.elementwise, args.autoaugment,
    #                                     args.cutmix,
    #                                     args.m, args.eta, args.norm_adaptive, args.seed,'_random{}'.format(args.random_idxs_frac) if args.random_idxs_frac>0. else '',
    # '_onlyBN' if args.only_bn else '', '_noBN' if args.no_bn else '', '_normalizeBias' if args.normalize_bias else '')
    model_name='_'.join([str(k)+'_'+str(v) for k, v in args._get_kwargs()]).replace('/','-')
    prefix = datetime.now().strftime("%y-%m-%d_%H:%M:%S/")+model_name
    base_folder = args.save+'/runs/' + prefix
    writer = SummaryWriter(base_folder)
    # Data Loader
    train_loader, test_loader = load_cifar(eval(args.dataset), args.m, autoaugment=args.autoaugment, data_path=args.data_path, random_idxs_frac=args.random_idxs_frac)
    num_classes = 10 if args.dataset == 'CIFAR10' else 100

    print('Creating Model...')
    # Model
    if args.model == 'pyramid':
        model = PyramidNet('cifar100', depth=272, alpha=200, num_classes=num_classes, bottleneck=True)
    elif 'PreAct' in args.model:
        model=eval(args.model)(num_classes).cuda()
        model.apply(init_weights(args.model))
    elif 'ConvMixer' in args.model:
        model = eval(args.model)(dim=256, depth=8, kernel_size=8, patch_size=1, n_classes=num_classes).cuda()
        print(sum([p.numel() for p in model.parameters()]))
    elif args.model=='vit_t':
        from timm.models.vision_transformer import VisionTransformer
        model=VisionTransformer(img_size=32,patch_size=4,num_classes=num_classes,embed_dim=192, depth=12, num_heads=3)
    elif args.model=='vit_s':
        from timm.models.vision_transformer import VisionTransformer
        # model=VisionTransformer(img_size=32,patch_size=4,num_classes=100, embed_dim=384, depth=12, num_heads=6)
        model=VisionTransformer(img_size=32,patch_size=4,num_classes=num_classes, embed_dim=384, depth=12, num_heads=6)
    else:
        model = eval(args.model)(num_classes=num_classes).cuda()

    print('Model created.')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(model)

    print('Putting model on device...')
    model.to(device)
    print('On device.')
    # Minimizer
    if args.base_minimizer=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.base_minimizer=='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.minimizer == 'SGD' or args.minimizer=='AdamW':
        minimizer = optimizer
    elif 'BN' in args.minimizer:
        minimizer = eval(args.minimizer)(optimizer, model, rho=args.rho, eta=args.eta, layerwise=args.layerwise,
                                         elementwise=args.elementwise, filterwise=args.filterwise, p=args.p,
                                         normalize_bias=args.normalize_bias, no_bn=args.no_bn, only_bn = args.only_bn, update_grad = args.update_grad, no_grad_norm=args.no_grad_norm)
    else:
        minimizer = eval(args.minimizer)(optimizer, model, rho=args.rho, eta=args.eta, layerwise=args.layerwise,
                                     elementwise=args.elementwise, filterwise=args.filterwise, p=args.p, normalize_bias=args.normalize_bias)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer if (args.minimizer=='SGD' or args.minimizer=='AdamW') else minimizer.optimizer, args.epochs)

    norm_adaptive = eval('attacks.weights.norms.'+args.norm_adaptive)()

    # Loss Functions
    if args.cutmix:
        print('Using CutMixCrossEntropyLoss')
        criterion = CutMixCrossEntropyLoss(True)
    elif args.smoothing:
        # raise NotImplementedError
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    print('Starting to train...')
    start_time = time.time()
    best_accuracy = 0.
    loss_best = 0.
    for epoch in range(args.epochs):
        # perform update step with base optimizer instead of SAM
        if args.start_sam<=epoch<args.end_sam:
            base_step=False
        else:
            base_step=True
        epoch_start = time.time()
        # Train
        model.train()
        loss = 0.
        loss_adv = 0.
        accuracy = 0.
        cnt = 0.
        for idx, (inputs, targets) in enumerate(train_loader):
            # print('Step, ',idx)
            unperturbed_model = deepcopy(model)
            inputs = inputs.to(device)
            targets = targets.type(torch.int64).to(device)
            if (epoch==0 and idx==0) or not args.ascent_with_old_batch: # use current batch for ascent step
                inputs_ascent, targets_ascent = inputs.clone(), targets.clone()
            # Ascent Step
            if 'Extra' in args.minimizer: # no forward-backward needed, since old gradient is used for ascent step
                batch_loss = torch.tensor(0.)
                if not (idx==0 and epoch==0): # no ascent step in first batch, no old gradient available yet
                    minimizer.ascent_step()
                    grads_1 = [minimizer.state[p]['old_grad'].data.clone().detach() for p in model.parameters() if p.grad is not None]
            elif 'AVG' in args.minimizer:
                batch_loss=torch.tensor(0.)
                minimizer.ascent_step()
                grads_1 = [p.grad.data.clone().detach() for p in model.parameters() if
                           p.grad is not None]
            else:
                predictions = model(inputs_ascent)
                batch_loss = criterion(predictions, targets_ascent)
                batch_loss.mean().backward()
                grads_1 = [p.grad.data.clone().detach() for p in model.parameters() if p.grad is not None]
                grads_1_bn = [p.grad.data.clone().detach() for n,p in model.named_parameters() if
                           (p.grad is not None and ('bn' in n or 'norm' in n))]
                grads_1_gamma = [p.grad.data.clone().detach() for n,p in model.named_parameters() if
                           (p.grad is not None and ('bn' in n or 'norm' in n) and 'weight' in n)]
                grads_1_nobn = torch.cat([p.grad.clone().detach().cpu().flatten() for n, p in model.named_parameters() if not
                                            ('norm' in n or 'bn' in n)]).numpy()
                if idx%100==0:
                    writer.add_scalar('grad/all',
                                      numpy.linalg.norm(torch.cat([t.cpu().flatten() for t in grads_1]).numpy()),
                                      global_step=epoch * len(train_loader) + idx)
                    writer.add_scalar('grad/noBN', numpy.linalg.norm(grads_1_nobn),
                                      global_step=epoch * len(train_loader) + idx)
                    writer.add_scalar('grad/onlyBN',
                                      numpy.linalg.norm(torch.cat([t.cpu().flatten() for t in grads_1_bn]).numpy()),
                                      global_step=epoch * len(train_loader) + idx)
                    writer.add_scalar('grad/all_max',
                                      numpy.linalg.norm(torch.max(
                                          torch.abs(torch.cat([t.cpu().flatten() for t in grads_1]))).numpy()),
                                      global_step=epoch * len(train_loader) + idx)
                    writer.add_scalar('grad/noBN_max', numpy.linalg.norm(numpy.max(numpy.abs(grads_1_nobn))))
                    writer.add_scalar('grad/onlyBN_max',
                                      numpy.linalg.norm(torch.max(
                                          torch.abs(torch.cat([t.cpu().flatten() for t in grads_1_bn]))).numpy()),
                                      global_step=epoch * len(train_loader) + idx)

                if (epoch+1)%50==0 and idx==len(train_loader)-1:
                    pass
                    # writer.add_histogram('grad/noBN', grads_1_nobn,global_step=epoch * len(train_loader) + idx)
                    # writer.add_histogram('grad/all', torch.cat([t.cpu().flatten() for t in grads_1]).numpy(),global_step=epoch * len(train_loader) + idx)
                    # writer.add_histogram('grad/onlyBN', torch.cat([t.cpu().flatten() for t in grads_1_bn]).numpy(),global_step=epoch * len(train_loader) + idx)

                if (args.minimizer =='SGD') or (args.minimizer =='AdamW'):
                    minimizer.step()
                    minimizer.zero_grad()
                elif base_step:
                    minimizer.optimizer.step()
                    minimizer.optimizer.zero_grad()
                else:
                    minimizer.ascent_step()
            norm = norm_adaptive(unperturbed_model, model, layers = [i for i in range(len(list(model.parameters())))]).item()
            if (args.minimizer !='SGD') and (args.minimizer !='AdamW') and not base_step:
                # Descent Step
                predictions_adv = model(inputs)
                batch_loss_2 = criterion(predictions_adv, targets)
                batch_loss_2.mean().backward()
                if 'Extra' in args.minimizer or 'AVG' in args.minimizer:
                    predictions=predictions_adv # for Extra-Gradient, no forward pass was executed for ascent step
                if not (idx==0 and epoch==0):
                    if idx % 50 == 0:
                        pass
                        # grads_2 = [p.grad.data.clone().detach() for n,p in model.named_parameters() if p.grad is not None]
                        # grads_2_bn = [p.grad.data.clone().detach() for n,p in model.named_parameters() if (p.grad is not None and ('bn' in n or 'norm' in n))]
                        # grads_2_gamma = [p.grad.data.clone().detach() for n,p in model.named_parameters() if (p.grad is not None and ('bn' in n or 'norm' in n) and 'weight' in n)]
                        # cosine = angle_between_grads(grads_1, grads_2)
                        # cosine_bn = angle_between_grads(grads_1_bn, grads_2_bn)
                        # cosine_gamma = angle_between_grads(grads_1_gamma, grads_2_gamma)
                        # writer.add_scalar('train/cosine_full', cosine,
                        #                   global_step=epoch * len(train_loader) + idx)
                        # writer.add_scalar('train/cosine_bn', cosine_bn,
                        #                   global_step=epoch * len(train_loader) + idx)
                        # writer.add_scalar('train/cosine_gamma', cosine_gamma,
                        #                   global_step=epoch * len(train_loader) + idx)
                    # if idx == len(train_loader) - 1:
                    #     gammas_adv = torch.cat([p.clone().detach().cpu() for n, p in model.named_parameters() if
                    #                         (('norm' in n or 'bn' in n) and ('weight' in n))]).numpy()
                    #     betas_adv = torch.cat([p.clone().detach().cpu() for n, p in model.named_parameters() if
                    #                        (('norm' in n or 'bn' in n) and ('bias' in n))]).numpy()
                    #     writer.add_histogram('params_all/BN_weights_adv', gammas_adv, global_step=epoch * len(train_loader) + idx)
                    #     writer.add_histogram('params_all/BN_bias_adv', betas_adv, global_step=epoch * len(train_loader) + idx)
                    #     for n,p in model.named_parameters():
                    #         if 'norm' in n:
                    #             gammas_adv = p.clone().detach().cpu().numpy()
                    #             writer.add_histogram('params_separate/{}'.format(n), gammas_adv,
                    #                                  global_step=epoch * len(train_loader) + idx)

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
                if (args.minimizer == 'SGD') or (args.minimizer == 'AdamW') or base_step:
                    loss_adv=0
                    batch_loss_2=torch.tensor(0.)
                else:
                    loss_adv += batch_loss_2.sum().item()
                if args.cutmix:
                    accuracy += (torch.argmax(predictions, 1) == torch.argmax(targets_ascent, 1)).sum().item()
                else:
                    accuracy += (torch.argmax(predictions, 1) == targets_ascent).sum().item()
            if args.ascent_with_old_batch:
                inputs_ascent, targets_ascent = inputs.clone(), targets.clone()  # re-use current batch for next ascent step
            if idx%50==0:
                writer.add_scalar('train/Batch_loss', batch_loss.mean().item(), global_step=epoch*len(train_loader)+idx)
                writer.add_scalar('train/Batch_loss_adversarial', batch_loss_2.mean().item(), global_step=(epoch)*len(train_loader)+idx)
                # writer.add_scalar('train/norm_adaptive', norm, global_step=epoch*len(train_loader)+idx)
            if idx==len(train_loader)-1:
                gammas = torch.cat([p.clone().detach().cpu() for n, p in model.named_parameters() if
                                    (('norm' in n or 'bn' in n) and ('weight' in n))]).numpy()
                betas = torch.cat([p.clone().detach().cpu() for n, p in model.named_parameters() if
                                    (('norm' in n or 'bn' in n) and ('bias' in n))]).numpy()
                writer.add_histogram('params_all/BN_bias', betas, global_step=epoch*len(train_loader)+idx)
                writer.add_histogram('params_all/BN_weights', gammas, global_step=epoch*len(train_loader)+idx)
                for n, p in model.named_parameters():
                    if 'norm' in n:
                        gammas = p.clone().detach().cpu().numpy()
                        writer.add_histogram('params_separate/{}'.format(n), gammas,
                                             global_step=epoch * len(train_loader) + idx)


            # print('Norm adaptive: ', norm)
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
        print(f"Epoch: {epoch}, Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}, Time: {time.time()-epoch_start}")
    end_time = time.time()
    state['runtime'] = end_time-start_time
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
    parser.add_argument("--minimizer", default='ASAM', type=str, help="ASAM, SAM or SGD, or Extra variants.")
    parser.add_argument("--base_minimizer", default='SGD', type=str, help="SGD or AdamW")
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
    parser.add_argument("--ascent_with_old_batch", action='store_true', help="perform ascent step with batch from previous step")
    parser.add_argument("--random_idxs_frac", default=0.0, type=float, help="set frac of train labels to random")
    parser.add_argument("--no_bn", action='store_true', help="perform ascent step without bn layer")
    parser.add_argument("--only_bn", action='store_true', help="perform ascent step only with bn layer")
    parser.add_argument("--update_grad", action='store_true', help="use momentum of gradient for sam update")
    parser.add_argument("--no_grad_norm", action='store_true', help="don't normalize gradient step (only usable with SAM_BN")
    parser.add_argument("--start_sam", default=0, type=int, help="start SAM at this epoch")
    parser.add_argument("--end_sam", default=100000, type=float, help="end SAM at this epoch")

    args = parser.parse_args()
    print(args.elementwise, args.filterwise, args.layerwise)
    print('cutmix: ', args.cutmix)
    assert args.dataset in ['CIFAR10', 'CIFAR100'], \
        f"Invalid data type. Please select CIFAR10 or CIFAR100"
    assert args.minimizer in ['ASAM', 'SAM', 'SGD', 'AdamW','ExtraASAM', 'ExtraSAM', 'SAM_BN', 'ASAM_BN', 'AVG_ASAM_BN', 'AVG_SAM_BN', 'SAM_BN_WEIGHTS', 'ASAM_BN_WEIGHTS', 'ASAM_BN_FC', 'SAM_BN_FC', 'FISHER_SAM'], \
        f"Invalid minimizer type."
    assert args.start_sam<args.end_sam

    # set seed
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)
    # Make save directory
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)
    train(args)
