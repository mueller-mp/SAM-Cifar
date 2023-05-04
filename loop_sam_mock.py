'''
script to loop over hyperparameters and submit many batch scripts to slurm
'''
import os
loop = 'loop.job'

params=[
    ('SGD', 'L2Norm', '2', False, False, False, 0.05, 0.),  # SGD
    ('SAM', 'L2Norm', '2', False, False, False, 0.2, 0.),  # SAM
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.2, 0.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.2, 1.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.2, 2.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.2, 3.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.2, 4.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.2, 5.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.2, 10.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.02, 0.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.02, 1.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.02, 2.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.02, 3.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.02, 4.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.02, 5.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.02, 10.),  # SAM mock
]

params_2=[
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.2, 20.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.2, 50.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.002, 0.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.002, 1.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.002, 2.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.002, 3.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.002, 4.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.002, 5.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.002, 10.),  # SAM mock
]

params_3=[
('SAM_mock', 'L2Norm', '2', False, False, False, 0.02, 20.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.02, 50.),  # SAM mock
('SAM_mock', 'L2Norm', '2', False, False, False, 0.002, 20.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.002, 50.),  # SAM mock
('SAM_mock', 'L2Norm', '2', False, False, False, 2., 20.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 2., 50.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.0002, 0.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.0002, 1.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.0002, 2.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.0002, 3.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.0002, 4.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.0002, 5.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.0002, 10.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.0002, 20.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.0002, 50.),  # SAM mock
]

params_4=[
('SAM_mock', 'L2Norm', '2', False, False, False, 0.02, 100.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.02, 500.),  # SAM mock
('SAM_mock', 'L2Norm', '2', False, False, False, 0.002, 100.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.002, 500.),  # SAM mock
('SAM_mock', 'L2Norm', '2', False, False, False, .006, 20.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, .006, 50.),  # SAM mock
('SAM_mock', 'L2Norm', '2', False, False, False, .06, 20.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, .06, 50.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.0005, 0.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.0005, 1.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.0005, 2.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.005, 0.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.005, 1.),  # SAM mock
    ('SAM_mock', 'L2Norm', '2', False, False, False, 0.005, 2.),  # SAM mock
]

params_trades=[
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.0001, 1.),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.001, 1.),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.005, 1.),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.01, 1.),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.05, 1.),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.1, 1.),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.5, 1.),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.0001, 0.1),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.001, 0.1),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.005, 0.1),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.01, 0.1),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.05, 0.1),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.1, 0.1),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.5, 0.1),  # TradeSAM
]

params_trades_2=[
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.0001, 10.),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.005, 10.),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.001, 10.),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.005, 10.),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.0001, 5.),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.0001, 5.),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.005, 5.),  # TradeSAM
    ('TradeSAM', 'L2Norm', '2', False, False, False, 0.005, 5.),  # TradeSAM
]
import time
folder = 'TradeSAM'
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
    normalize_bias='--normalize_bias'
    eval_ascent = ''
    ascent_with_old_batch=''
    alpha=0.
    for minimizer, norm_adaptive, p, e, l, f, rho, beta in params_trades_2:
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
            fh.writelines('#SBATCH --time=0-10:00            # Runtime in D-HH:MM\n')
            fh.writelines(
                '#SBATCH --gres=gpu:1    # optionally type and number of gpus\n')
            fh.writelines(
                '#SBATCH --mem=50G                # Memory pool for all cores (see also --mem-per-cpu)\n')
            fh.writelines(
                '#SBATCH --output=/mnt/qb/work/hein/mmueller67/ASAM/logs/hostname_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME\n')
            fh.writelines(
                '#SBATCH --error=/mnt/qb/work/hein/mmueller67/ASAM/logs/hostname_%j.err   # File to which STDERR will be written - make sure this is not on $HOME\n')
            fh.writelines(
                '#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL\n')
            fh.writelines(
                '#SBATCH --mail-user=maximilian.mueller@wsii.uni-tuebingen.de   # Email to which notifications will be sent\n')
            fh.writelines('scontrol show job $SLURM_JOB_ID\n')
            fh.writelines(
                'python TradeSAM.py --dataset CIFAR100 --data_path {} --model {} --minimizer {} --lr {} --momentum {} --weight_decay {} --batch_size {} --epochs {} --smoothing {} --rho {} --beta {} --p {} {} {} {} {} {} --eta {} --m {} --save {} --norm_adaptive {} {} --seed {} --alpha {} {} {}'.format(data_path, model, minimizer, lr, momentum, weight_decay, batch_size, epochs, smoothing, rho, beta, p, layerwise, filterwise, elementwise, autoaugment, cutmix, eta, m, save, norm_adaptive, normalize_bias, seed, alpha, eval_ascent, ascent_with_old_batch))
        os.system('sbatch %s' % loop)
        time.sleep(1)  # Sleep for 1 second in order to minimize race conflicts for folder creation


