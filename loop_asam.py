'''
script to loop over hyperparameters and submit many batch scripts to slurm
'''
import os
loop = 'loop.job'

params_ASAM_eta_zero=[
    ('SGD', 'L2Norm', '2', False, False, False, 0.05),  # SGD
    ('SAM', 'L2Norm', '2', False, False, False, 0.01),  # SAM
    ('SAM', 'L2Norm', '2', False, False, False, 0.1),
    ('SAM', 'L2Norm', '2', False, False, False, 0.5),
    ('SAM', 'L2Norm', '2', False, False, False, 1.),
    ('SAM', 'L2Norm', '2', False, False, False, 5.),
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 0.01), # Elementwise l2
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 0.1),
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5),
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 1.),
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 5.),
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 10.),
    ('ASAM', 'LayerWiseL2NormAsam', '2', False, True, False, 0.01),  # Layerwise l2
    ('ASAM', 'LayerWiseL2NormAsam', '2', False, True, False, 0.1),
    ('ASAM', 'FilterWiseL2NormAsam', '2', False, False, True, 0.01),  # Filterwise l2
    ('ASAM', 'FilterWiseL2NormAsam', '2', False, False, True, 0.1),
    ('ASAM', 'FilterWiseL2NormAsam', '2', False, False, True, 0.5),
    ('ASAM', 'FilterWiseL2NormAsam', '2', False, False, True, 1.),
    ('ASAM', 'FilterWiseL2NormAsam', '2', False, False, True, 2.5),
    ('ASAM', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.00001),  # Elementwise linf
    ('ASAM', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.0001),
    ('ASAM', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.001),
    ('ASAM', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.01),
    ('ASAM', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.05),
    ('ASAM', 'LayerwiseLinfNormAsam', 'infinity', False, True, False, 0.00001),  # Layerwise linf
    ('ASAM', 'LayerwiseLinfNormAsam', 'infinity', False, True, False, 0.0001),
    ('ASAM', 'LayerwiseLinfNormAsam', 'infinity', False, True, False, 0.001),
    ('ASAM', 'LayerwiseLinfNormAsam', 'infinity', False, True, False, 0.01),
    ('ASAM', 'LayerwiseLinfNormAsam', 'infinity', False, True, False, 0.05),
    ('ASAM', 'FilterwiseLinfNormAsam', 'infinity', False, False, True, 0.00001),  # Filterwise linf
    ('ASAM', 'FilterwiseLinfNormAsam', 'infinity', False, False, True, 0.0001),
    ('ASAM', 'FilterwiseLinfNormAsam', 'infinity', False, False, True, 0.001),
    ('ASAM', 'FilterwiseLinfNormAsam', 'infinity', False, False, True, 0.01),
    ('ASAM', 'FilterwiseLinfNormAsam', 'infinity', False, False, True, 0.05),
]


params_ASAM_eta_zero_2=[
    ('SAM', 'L2Norm', '2', False, False, False, 0.2),  # SAM
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 10.), # Elementwise l2
    ('ASAM', 'LayerWiseL2NormAsam', '2', False, True, False, 0.03),  # Layerwise l2
    ('ASAM', 'LayerWiseL2NormAsam', '2', False, True, False, 0.001),
    ('ASAM', 'LayerWiseL2NormAsam', '2', False, True, False, 0.02),
    ('ASAM', 'LayerWiseL2NormAsam', '2', False, True, False, 0.05),
    ('ASAM', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 5e-3),  # Elementwise linf
    ('ASAM', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.02),
    ('ASAM', 'LayerwiseLinfNormAsam', 'infinity', False, True, False, 1e-6),  # Layerwise linf
    ('ASAM', 'LayerwiseLinfNormAsam', 'infinity', False, True, False, 5e-6),
    ('ASAM', 'LayerwiseLinfNormAsam', 'infinity', False, True, False, 1e-7),
    ('ASAM', 'FilterwiseLinfNormAsam', 'infinity', False, False, True, 5e-4),  # Filterwise linf
]

params_ASAM_eta_zero_3=[
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 2.), # Elementwise l2
    ('ASAM', 'LayerwiseLinfNormAsam', 'infinity', False, True, False, 2e-6),  # Layerwise linf
    ('ASAM', 'LayerwiseLinfNormAsam', 'infinity', False, True, False, 6e-7),
    ('ASAM', 'FilterwiseLinfNormAsam', 'infinity', False, False, True, 5e-5),  # Filterwise linf
    ('ASAM', 'FilterwiseLinfNormAsam', 'infinity', False, False, True, 2e-4),
]

params_ASAM_eta_test=[
    ('SAM', 'L2Norm', '2', False, False, False, 0.01),  # SAM
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 0.01), # Elementwise l2
    ('ASAM', 'LayerWiseL2NormAsam', '2', False, True, False, 0.01),  # Layerwise l2
    ('ASAM', 'FilterWiseL2NormAsam', '2', False, False, True, 0.01),  # Filterwise l2
    ('ASAM', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.00001),  # Elementwise linf
    ('ASAM', 'LayerwiseLinfNormAsam', 'infinity', False, True, False, 0.00001),  # Layerwise linf
    ('ASAM', 'FilterwiseLinfNormAsam', 'infinity', False, False, True, 0.00001),  # Filterwise linf
]

params_eval_ascent = [
    ('SAM', 'L2Norm', '2', False, False, False, 0.001),  # SAM
    ('SAM', 'L2Norm', '2', False, False, False, 0.01),
    ('SAM', 'L2Norm', '2', False, False, False, 0.1),
    ('SAM', 'L2Norm', '2', False, False, False, 0.5),
    ('SAM', 'L2Norm', '2', False, False, False, 1.),
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 0.01),  # Elementwise l2
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 0.1),
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5),
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 1.),
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 5.),
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 10.),
]

params_eval_ascent_2 = [
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 1.),  # Elementwise l2
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 2.),
    ('ASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 3.),
]

params_Extra = [
    ('ExtraSAM', 'L2Norm', '2', False, False, False, 0.001),  # SAM
    ('ExtraSAM', 'L2Norm', '2', False, False, False, 0.01),
    ('ExtraSAM', 'L2Norm', '2', False, False, False, 0.1),
    ('ExtraSAM', 'L2Norm', '2', False, False, False, 0.5),
    ('ExtraSAM', 'L2Norm', '2', False, False, False, 1.),
    ('ExtraASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 0.01),  # Elementwise l2
    ('ExtraASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 0.1),
    ('ExtraASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5),
    ('ExtraASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 1.),
    ('ExtraASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 5.),
    ('ExtraASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 10.),
    ('ExtraASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 1.),  # Elementwise l2
    ('ExtraASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 2.),
    ('ExtraASAM', 'ElementWiseL2NormAsam', '2', True, False, False, 3.),
]

import time
folder = 'ascent_with_old_batch'
for seed in range(1):
    model='wrn28_10'
    lr=0.1
    momentum=0.9
    weight_decay=5e-4
    batch_size=128
    epochs=100
    smoothing=0.
    autoaugment='' # --autoaugment
    cutmix='' # --cutmix
    eta=0.0
    m=128
    save='./snapshots/'+folder
    data_path = '/mnt/qb/hein/datasets/CIFAR100'
    normalize_bias=''
    eval_ascent = ''
    ascent_with_old_batch='--ascent_with_old_batch'
    for minimizer, norm_adaptive, p, e, l, f, rho in params_eval_ascent+params_eval_ascent_2:  #(params_ASAM_eta_zero+params_ASAM_eta_zero_2+params_ASAM_eta_zero_3):

        elementwise = '--elementwise' if e else ''
        layerwise = '--layerwise' if l else ''
        filterwise = '--filterwise' if f else ''

        with open(loop, 'w+') as fh:
            fh.writelines('#!/bin/bash\n')
            fh.writelines(
                '#SBATCH --ntasks=1                # Number of tasks (see below)\n')
            fh.writelines(
                '#SBATCH --cpus-per-task=4       # Number of CPU cores per task\n')
            fh.writelines(
                '#SBATCH --nodes=1                 # Ensure that all cores are on one machine\n')
            fh.writelines('#SBATCH --time=0-06:00            # Runtime in D-HH:MM\n')
            fh.writelines(
                '#SBATCH --gres=gpu:1    # optionally type and number of gpus\n')
            fh.writelines(
                '#SBATCH --mem=50G                # Memory pool for all cores (see also --mem-per-cpu)\n')
            fh.writelines(
                '#SBATCH --output=/mnt/qb/hein/mmueller67/ASAM/logs/hostname_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME\n')
            fh.writelines(
                '#SBATCH --error=/mnt/qb/hein/mmueller67/ASAM/logs/hostname_%j.err   # File to which STDERR will be written - make sure this is not on $HOME\n')
            fh.writelines(
                '#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL\n')
            fh.writelines(
                '#SBATCH --mail-user=maximilian.mueller@wsii.uni-tuebingen.de   # Email to which notifications will be sent\n')
            fh.writelines('scontrol show job $SLURM_JOB_ID\n')
            fh.writelines(
                'python ExtraASAM.py --dataset CIFAR100 --data_path {} --model {} --minimizer {} --lr {} --momentum {} --weight_decay {} --batch_size {} --epochs {} --smoothing {} --rho {} --p {} {} {} {} {} {} --eta {} --m {} --save {} --norm_adaptive {} {} --seed {} {} {}'.format(data_path, model, minimizer, lr, momentum, weight_decay, batch_size, epochs, smoothing, rho, p, layerwise, filterwise, elementwise, autoaugment, cutmix, eta, m, save, norm_adaptive, normalize_bias, seed, eval_ascent, ascent_with_old_batch))
        os.system('sbatch %s' % loop)
        time.sleep(1)  # Sleep for 1 second in order to minimize race conflicts for folder creation


