'''
script to loop over hyperparameters and submit many batch scripts to slurm
'''
import os
loop = 'loop.job'

params=[
    ('SGD', 'L2Norm', '2', False, False, False, 0.2),  # SGD
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.2),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.5),  # SAM
    # ('SAM_BN', 'L2Norm', '2', False, False, False, 0.05),  # SAM
    # ('SAM_BN', 'L2Norm', '2', False, False, False, 0.1),  # SAM
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 1.), # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5.), # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 10.), # Elementwise l2
    # ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, .5), # Elementwise l2
    # ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 5e-3),  # Elementwise linf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.02),
    # ('ASAM_BN', 'LayerwiseLinfNormAsam', 'infinity', False, True, False, 1e-2),
]

params_2=[
# ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 1e-3),  # Elementwise linf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.05),
    ('SAM_BN', 'L2Norm', '2', False, False, False, 1.),  # SAM
    # ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, .25),  # Elementwise l2
]

params_3 = [
    ('ASAM_BN', 'LayerWiseL2NormAsam', '2', False, True, False, 1e-2),
    # ('ASAM_BN', 'LayerWiseL2NormAsam', '2', False, True, False, 3e-2),
    # ('ASAM_BN', 'LayerWiseL2NormAsam', '2', False, True, False, 6e-3),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.1),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.3),
    ('SAM_BN', 'L2Norm', '2', False, False, False, 3.),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 5.),  # SAM
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 2.),  # Elementwise l2
]

params_4 = [
    ('ASAM_BN', 'LayerWiseL2NormAsam', '2', False, True, False, 0.1),
    ('ASAM_BN', 'LayerWiseL2NormAsam', '2', False, True, False, 0.3),
]

params_5 = [
    ('ASAM_BN', 'LayerWiseL2NormAsam', '2', False, True, False, 0.5),
    # ('ASAM_BN', 'LayerWiseL2NormAsam', '2', False, True, False, 1.),
]

# params_6 = [
#     ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 50.), # Elementwise l2
#     ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 100.),  # Elementwise l2
# ]

# params_7 = [
#     ('ASAM_BN', 'FilterWiseL2NormAsam', '2', False, False, True, .3),
#     ('ASAM_BN', 'FilterWiseL2NormAsam', '2', False, False, True, 1.),
#     ('ASAM_BN', 'FilterWiseL2NormAsam', '2', False, False, True, 5.),
#     ('ASAM_BN', 'FilterWiseL2NormAsam', '2', False, False, True, 30.),
#     ('ASAM_BN', 'FilterwiseLinfNormAsam', 'infinity', False, False, True, 1e-5),
#     ('ASAM_BN', 'FilterwiseLinfNormAsam', 'infinity', False, False, True, 1e-4),
#     ('ASAM_BN', 'FilterwiseLinfNormAsam', 'infinity', False, False, True, 1e-3),
# ]

params_8=[
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 20.),  # Elementwise l2
    # ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 50.),  # Elementwise l2
    # ('SAM_BN', 'L2Norm', '2', False, False, False, 0.05),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.1),  # SAM
    ('ASAM_BN', 'LayerWiseL2NormAsam', '2', False, True, False, 0.0001),
    ('ASAM_BN', 'LayerWiseL2NormAsam', '2', False, True, False, 0.0005),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.001),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.0005),

]

params_9 = [
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.01),  # SAM
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.0002),
]

params_10 = [
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.1),  # Elementwise l2
]

params_no_grad_norm = [
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.01),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, .05),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, .1),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, .5),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 1.),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 5.),  # SAM
]

params_no_grad_norm_2 = [
    ('SAM_BN', 'L2Norm', '2', False, False, False, .25),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 1.),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 2.5),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 10.),  # SAM
]

params_no_grad_norm_3 = [
    ('SAM_BN', 'L2Norm', '2', False, False, False, 10.),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 20.),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 30.),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 50.),  # SAM
]

params_no_grad_norm_4 = [
    ('SAM_BN', 'L2Norm', '2', False, False, False, 100.),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 300.),  # SAM
]

params_preact=[
('SGD', 'L2Norm', '2', False, False, False, 0.01),  # SAM
('SAM_BN', 'L2Norm', '2', False, False, False, 0.01),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.05),  # SAM
('SAM_BN', 'L2Norm', '2', False, False, False, 0.1),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.5),  # SAM
('SAM_BN', 'L2Norm', '2', False, False, False, 1.),  # SAM
]

params_preact_2=[
('SAM_BN', 'L2Norm', '2', False, False, False, 0.25),  # SAM
('SAM_BN', 'L2Norm', '2', False, False, False, 2.5),  # SAM
('SAM_BN', 'L2Norm', '2', False, False, False, 5.),  # SAM
# ('SAM_BN', 'L2Norm', '2', False, False, False, 10.),  # SAM
]

params_preact_3=[
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.1),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 1.),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 2.5),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5.),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 10.),  # Elementwise l2
]

params_preact_4 = [
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 12.5),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 15.),  # Elementwise l2

]

params_preact_el_inf = [
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.0001),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.0005),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.001),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.005),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.01),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.05),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.1),
]


params_preact_el_inf_2 = [
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.25),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.5),
]

params_asam_original=[
    # ('SGD', 'L2Norm', '2', False, False, False, 0.2),  # SGD
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.1),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.5),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 1.),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 5.),  # SAM
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 1.), # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5.), # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 10.), # Elementwise l2
]

params_asam_original_2=[
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.25),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 2.5),  # SAM
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 2.5), # Elementwise l2
]

params_test_samdavda=[
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.1),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.2),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.5),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 1.),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 5.),  # SAM
]


params_test_samdavda_2=[
    ('SAM_BN', 'L2Norm', '2', False, False, False, 10.),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 7.5),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 2.5),  # SAM
]

params_AVG = [
    ('AVG_SAM_BN', 'L2Norm', '2', False, False, False, 0.01),  # SAM
    ('AVG_SAM_BN', 'L2Norm', '2', False, False, False, 0.1),  # SAM
    ('AVG_SAM_BN', 'L2Norm', '2', False, False, False, 0.5),  # SAM
    ('AVG_SAM_BN', 'L2Norm', '2', False, False, False, 1.),  # SAM
    ('AVG_SAM_BN', 'L2Norm', '2', False, False, False, 5.),  # SAM
    ('AVG_SAM_BN', 'L2Norm', '2', False, False, False, 10.),  # SAM
    ('AVG_SAM_BN', 'L2Norm', '2', False, False, False, 50.),  # SAM
    ('AVG_SAM_BN', 'L2Norm', '2', False, False, False, 100.),  # SAM
    ('AVG_SAM_BN', 'L2Norm', '2', False, False, False, 500.),  # SAM
    ('AVG_SAM_BN', 'L2Norm', '2', False, False, False, 1000.),  # SAM
('SGD', 'L2Norm', '2', False, False, False, 0.5),  # SGD
]

params_asam_long=[
    ('SGD', 'L2Norm', '2', False, False, False, 0.5),  # SGD
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.05),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.1),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.5),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 1.),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 5.),  # SAM
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 1.),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 2.5),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5.),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 10.),  # Elementwise l2
    # ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 15.),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.01),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.001),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.1),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.2),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.5),  # Layerwise l2
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.0001), # elementwise L inf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.001), # elementwise L inf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.01), # elementwise L inf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.05), # elementwise L inf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.1), # elementwise L inf
]

params_asam_long_biasnorm=[
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.2),  # elementwise L inf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.5),  # elementwise L inf
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 1.),  # Layerwise l2
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.2),  # SAM
]

params_asam_long_nobiasnorm=[
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 1.),  # Layerwise l2
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.2),  # SAM
]

params_resnext = [
    # ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.05),  # Layerwise l2
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.05),  # elementwise L inf
]

params_resnet56 = [
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.05),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 1.),  # Layerwise l2
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.05),  # elementwise L inf
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.3),  # SAM
]

params_resnet56_2 = [
    # ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5),  # elem l2
    # ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.1),  # elem l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False, 0.05),  # elem inf
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.1),  # SAM
]

params_densenet = [
    # ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.5),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.05),  # Layerwise l2
    # ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.005),  # elementwise L inf
    # ('ASAM_BN', 'ElementwiseLinfNormAsam', '2', True, False, False, 0.5),  # elementwise L 2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5.),  # elementwise L 2
]

params_preact_new = [
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 1.),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5.),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 10.),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 15.),  # Elementwise l2
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.001),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.005),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.01),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.05),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.1),
]
asam_preact_sam = [
('SAM_BN', 'L2Norm', '2', False, False, False, 0.01),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.05),  # SAM
('SAM_BN', 'L2Norm', '2', False, False, False, 0.1),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.5),  # SAM
('SAM_BN', 'L2Norm', '2', False, False, False, 1.),  # SAM
('SAM_BN', 'L2Norm', '2', False, False, False, 5.),  # SAM
]

asam_preact_new_2=[
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.0001),
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.0005),
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.1),  # Elementwise l2
]
params_cifar10=[
    ('SGD', 'L2Norm', '2', False, False, False, 0.5, ''),  # SGD
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.05, ''),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.1, ''),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.5, ''),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.1, '--only_bn'),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.5, '--only_bn'),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 1., '--only_bn'),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 5., '--only_bn'),  # SAM
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.001, ''), # elementwise Linf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.005, ''),# elementwise Linf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.01, ''),# elementwise Linf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.05, ''),# elementwise Linf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.005, '--only_bn'),  # elementwise Linf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.01, '--only_bn'),  # elementwise Linf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.05, '--only_bn'),  # elementwise Linf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.1, '--only_bn'),  # elementwise Linf
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.005, ''),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.01, ''),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.05, ''),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.1, ''),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.5, ''),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.05, '--only_bn'),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.1, '--only_bn'),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.5, '--only_bn'),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 1., '--only_bn'),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5, ''),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 1., ''),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5., ''),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 10., ''),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5, '--only_bn'),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 1., '--only_bn'),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5., '--only_bn'),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 10., '--only_bn'),  # Elementwise l2
]

params_cifar10_2=[
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.25, '--only_bn'),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5.5, '--only_bn'),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.25, ''),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5.5, ''),  # Elementwise l2
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.25, ''),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 2.5, '--only_bn'),  # SAM
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.25, '--only_bn'),  # Layerwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.025, ''),  # Layerwise l2
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.025, '--only_bn'),  # elementwise Linf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.25, '--only_bn'),  # elementwise Linf
]

params_cifar10_3=[
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 2., '--only_bn'),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 3., '--only_bn'),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 3., ''),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 2., ''),  # Elementwise l2
]
params_cifar10_4=[
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 2., '--only_bn'),  # Elementwise l2
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.05, '--only_bn'),  # elementwise Linf
]

params_track_weights=[
    ('SGD', 'L2Norm', '2', False, False, False, 0.5, ''),  # SGD
    ('SAM_BN', 'L2Norm', '2', False, False, False, 0.2, ''),  # SAM
    ('SAM_BN', 'L2Norm', '2', False, False, False, 1., '--only_bn'),  # SAM
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5., '--only_bn'),  # Elementwise l2
    ('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5., ''),  # Elementwise l2
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.05, '--only_bn'),  # elementwise Linf
    ('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.01, ''),  # elementwise Linf
]


params_rho2asam=[
    ('SAM_2RHO', 'ElementwiseLinfNormAsam', '2', False, False, False, 1., 0.1),  # SAM
    # ('SAM_2RHO', 'ElementwiseLinfNormAsam', '2', False, False, False, 0.1, 0.5),  # SAM
    ('ASAM_2RHO', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.05, 0.01 ),  # elementwise Linf
    # ('ASAM_2RHO', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.1, 0.0000001),  # elementwise Linf
    # ('ASAM_2RHO', 'ElementWiseL2NormAsam', '2', False, True, False, 0.5, 0.05),  # Layerwise l2
    ('ASAM_2RHO', 'ElementWiseL2NormAsam', '2', False, True, False, 0.2, 0.02),  # Layerwise l2
]

import time
folder = 'RHO2ASAM_2' # 'ASAM_BN_Cifar10' # resnext_nosequential
for seed in range(1):
    # 'PreActResNet34BatchNorm', 'PreActResNet34LayerNorm' 'PreActResNet34GroupNorm' PreActResNet34LayerNormSmall
    # ConvMixerGroup, ConvMixerBatch, ConvMixerLayer
    for model in ['wrn28_10']: #['wrn28_10', 'resnext29_32x4d_nosequential', 'resnet56_nosequential', 'densenet100_nosequential']: #['PreActResNet34LayerNorm', 'PreActResNet34GroupNorm', 'PreActResNet34LayerNormSmall']: #resnext29_32x4d_nosequential #resnet56_nosequential #densenet100_nosequential
        for autoaugment in ['--autoaugment']:  #--autoaugment
            dataset = 'CIFAR100'
            # model = 'densenet100_nosequential' #
            no_grad_norm=''
            lr=0.1
            momentum=0.9
            weight_decay=5e-4
            epochs=200
            smoothing=0.1
            cutmix='' # --cutmix
            eta=0.0
            batch_size=128
            m=batch_size
            save='./snapshots/'+folder
            data_path = '/mnt/qb/hein/datasets/'+dataset
            normalize_bias='--normalize_bias' #'--normalize_bias'
            eval_ascent = ''
            ascent_with_old_batch=''
            no_bn=''
            only_bn=''
            random_idxs_frac=0.
            for minimizer, norm_adaptive, p, e, l, f, rho_bn, rho_conv in params_rho2asam: #params+params_2+params_3+params_4+params_5+params_8+params_9+params_10: #params+params_2+params_3+params_4+params_5+params_8+params_9:
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
                    fh.writelines('#SBATCH --time=0-23:00            # Runtime in D-HH:MM\n')
                    fh.writelines(
                        '#SBATCH --gres=gpu:1    # optionally type and number of gpus\n')
                    fh.writelines(
                        '#SBATCH --mem=50G                # Memory pool for all cores (see also --mem-per-cpu)\n')
                    fh.writelines(
                        '#SBATCH --output=/mnt/qb/work/hein/mmueller67/logs/hostname_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME\n')
                    fh.writelines(
                        '#SBATCH --error=/mnt/qb/work/hein/mmueller67/logs/hostname_%j.err   # File to which STDERR will be written - make sure this is not on $HOME\n')
                    fh.writelines(
                        '#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL\n')
                    fh.writelines(
                        '#SBATCH --mail-user=maximilian.mueller@wsii.uni-tuebingen.de   # Email to which notifications will be sent\n')
                    fh.writelines('scontrol show job $SLURM_JOB_ID\n')
                    fh.writelines(
                        'python train_RHO2ASAM.py --dataset {} --data_path {} --model {} --minimizer {} --lr {} --momentum {} --weight_decay {} --batch_size {} --epochs {} --smoothing {} --rho_bn {} --rho_conv {} --p {} {} {} {} {} {} --eta {} --m {} --save {} --norm_adaptive {} {} --seed {} --random_idxs_frac {} {} {} {} {} {}'.format(dataset, data_path, model, minimizer, lr, momentum, weight_decay, batch_size, epochs, smoothing, rho_bn, rho_conv, p, layerwise, filterwise, elementwise, autoaugment, cutmix, eta, m, save, norm_adaptive, normalize_bias, seed, random_idxs_frac, eval_ascent, ascent_with_old_batch, only_bn, no_bn, no_grad_norm))
                os.system('sbatch -p gpu-v100 %s' % loop)
                time.sleep(2)  # Sleep for 2 seconds in order to minimize race conflicts for folder creation


