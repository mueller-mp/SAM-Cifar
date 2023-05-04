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
    # ('SGD', 'L2Norm', '2', False, False, False, 0.5, ''),  # SGD
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

params_track_weights_grad=[
    ('SGD', 'L2Norm', '2', False, False, False, 0.5, ''),  # SGD
    ('SAM_BN', 'L2Norm', '2', False, False, False, 1., '--only_bn'),  # SAM
]

params_only_one_type=[
    ('ASAM_BN_FC', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.05, ''),  # elementwise Linf
    ('ASAM_BN_FC', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.01, ''),  # elementwise Linf
    ('ASAM_BN_FC', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.1, ''),  # elementwise Linf
    ('ASAM_BN_FC', 'ElementWiseL2NormAsam', '2', True, False, False, 1., ''),  # Elementwise l2
    ('ASAM_BN_FC', 'ElementWiseL2NormAsam', '2', True, False, False, 5., ''),  # Elementwise l2
    ('ASAM_BN_FC', 'ElementWiseL2NormAsam', '2', True, False, False, 10., ''),  # Elementwise l2
    # ('ASAM_BN_WEIGHTS', 'ElementWiseL2NormAsam', '2', True, False, False, 1., '--only_bn'),  # Elementwise l2
    # ('ASAM_BN_WEIGHTS', 'ElementWiseL2NormAsam', '2', True, False, False, 5., '--only_bn'),  # Elementwise l2
    # ('ASAM_BN_WEIGHTS', 'ElementWiseL2NormAsam', '2', True, False, False, 10., '--only_bn'),  # Elementwise l2
    # ('ASAM_BN_WEIGHTS', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.01, '--only_bn'),  # elementwise Linf
    # ('ASAM_BN_WEIGHTS', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.1, '--only_bn'),  # elementwise Linf
    # ('ASAM_BN_WEIGHTS', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.1, '--only_bn'),  # elementwise Linf
]

params_cifar100_fillup=[
    # folder, model, minimizer, norm, p, elementwise, layerwise, filterwise, rho, normalizeBias, AA, onlybn
#wrn 28-10
#     ('ASAM_grid_long', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5, '', '', ''),  # elementwise l2
#     ('ASAM_grid_long_aa', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5, '', '--autoaugment', ''),  # elementwise l2
#     ('ASAM_grid_long', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5, '--normalize_bias', '', ''), # elementwise l2
#     ('ASAM_grid_long_aa', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5, '--normalize_bias', '--autoaugment', ''),  # elementwise l2
#
#     ('ASAM_grid_long', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False, 0.005, '', '', ''),# elementwise l2
#     ('ASAM_grid_long_aa', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False, 0.005, '', '--autoaugment', ''),  # elementwise l2
#     ('ASAM_grid_long', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False, 0.005, '--normalize_bias', '', ''),  # elementwise l2
#     ('ASAM_grid_long_aa', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False, 0.005, '--normalize_bias', '--autoaugment', ''),  # elementwise l2
#
#     ('ASAM_grid_long', 'wrn28_10', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False, 0.05, '', '', ''),  # elementwise l2
#     ('ASAM_grid_long_aa', 'wrn28_10', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False, 0.05, '', '--autoaugment', ''),  # elementwise l2
#     ('ASAM_grid_long', 'wrn28_10', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False, 0.05,  '--normalize_bias', '', ''),  # elementwise l2
#     ('ASAM_grid_long_aa', 'wrn28_10', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False, 0.05, '--normalize_bias', '--autoaugment', ''),  # elementwise l2
# #resnext
#     ('resnext_nosequential', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5, '--normalize_bias', '', ''),  # elementwise l2
#     ('resnext_nosequential_aa', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.5, '--normalize_bias', '--autoaugment', ''),  # elementwise l2
#  # densenet
#     ('densenet100_nosequential_aa', 'densenet100_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False,False, 0.1, '--normalize_bias', '--autoaugment', '--only_bn'),  # elementwise l2
    (
    'resnext_nosequential', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False,
    0.05, '--normalize_bias', '', ''),  # elementwise l2
]
params_cifar10_fillup=[
    ('ASAM_BN_Cifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False, 0.05, '--normalize_bias', '--autoaugment', '--only_bn'),  # elementwise l2
]


params_track_weights_grad=[
    ('TrackBNWeights_grad_2', 'SGD', 'L2Norm', '2', False, False, False, 0.5, ''),  # SGD
    ('TrackBNWeights_grad_2', 'SAM_BN', 'L2Norm', '2', False, False, False, 1., '--only_bn'),  # SAM
    ('TrackBNWeights_grad_2', 'ASAM_BN', 'L2Norm', '2', True, False, False, 1., ''),  # SAM
]

params_no_grad_norm_cifar10=[
    ('SAM_BN_NO_GRADNORM_LONG', 'SAM_BN', 'L2Norm', '2', False, False, False, '--only_bn'),  # SAM
]



# dataset, folder, model, minimizer, norm_adaptive, p, e, l, f, rho, eta, normalize_bias, autoaugment, only_bn,
params_more_seeds = [  # CIFAR10 RHO FIXED
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, 2.0, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, 3.0, 0.0, True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 0.5, 0.01, False, '', ''],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 0.5, 0.01, False, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True,
     False, False, 0.005, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True,
     False, False, 0.025, 0.0, True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True,
     False, 0.025, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True,
     False, 0.25, 0.0, True, '', '--only_bn'],
    ['CIFAR10', 'Fisher', 'densenet100_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.1, 1.0, True, '', ''],
    ['CIFAR10', 'Fisher', 'densenet100_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.1, 1.0, True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.1, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.5, 0.0, True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'SGD', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.5, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, 2.0, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, 3.0, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 0.5, 0.01, False, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 0.5, 0.01, False, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True,
     False, False, 0.005, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True,
     False, False, 0.025, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True,
     False, 0.025, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True,
     False, 0.25, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'Fisher', 'densenet100_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.1, 1.0, True, '--autoaugment', ''],
    ['CIFAR10', 'Fisher', 'densenet100_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.1, 1.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.1, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.5, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'SGD', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.5, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False,
     2.0, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False,
     3.0, 0.0, True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 0.5, 0.01, False, '', ''],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 0.5, 0.01, False, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False,
     False, 0.005, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False,
     False, 0.025, 0.0, True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False,
     0.025, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False,
     0.25, 0.0, True, '', '--only_bn'],
    ['CIFAR10', 'Fisher', 'resnet56_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1,
     1.0, True, '', ''],
    ['CIFAR10', 'Fisher', 'resnet56_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1,
     1.0, True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.1, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.5, 0.0, True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'SGD', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.5, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False,
     2.0, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False,
     3.0, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 0.5, 0.01, False, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 0.5, 0.01, False, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False,
     False, 0.005, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False,
     False, 0.025, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False,
     0.025, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False,
     0.25, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'Fisher', 'resnet56_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1,
     1.0, True, '--autoaugment', ''],
    ['CIFAR10', 'Fisher', 'resnet56_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1,
     1.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.1, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.5, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnet56_nosequential', 'SGD', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.5, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, 2.0, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, 3.0, 0.0, True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam',
     '2', True, False, False, 0.5, 0.01, False, '', ''],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam',
     '2', True, False, False, 0.5, 0.01, False, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True,
     False, False, 0.005, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True,
     False, False, 0.025, 0.0, True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True,
     False, 0.025, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True,
     False, 0.25, 0.0, True, '', '--only_bn'],
    ['CIFAR10', 'Fisher', 'resnext29_32x4d_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.1, 1.0, True, '', ''],
    ['CIFAR10', 'Fisher', 'resnext29_32x4d_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.1, 1.0, True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.1, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.5, 0.0, True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'SGD', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.5, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, 2.0, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, 3.0, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam',
     '2', True, False, False, 0.5, 0.01, False, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam',
     '2', True, False, False, 0.5, 0.01, False, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True,
     False, False, 0.005, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True,
     False, False, 0.025, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True,
     False, 0.025, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True,
     False, 0.25, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'Fisher', 'resnext29_32x4d_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.1, 1.0, True, '--autoaugment', ''],
    ['CIFAR10', 'Fisher', 'resnext29_32x4d_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.1, 1.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.1, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.5, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'resnext29_32x4d_nosequential', 'SGD', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.5, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 2.0, 0.0,
     True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 3.0, 0.0,
     True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, 0.5, 0.01, False, '', ''],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, 0.5, 0.01, False, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False,
     0.005, 0.0, True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False,
     0.025, 0.0, True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.025, 0.0,
     True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.25, 0.0,
     True, '', '--only_bn'],
    ['CIFAR10', 'Fisher', 'wrn28_10', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1, 1.0, True,
     '', ''],
    ['CIFAR10', 'Fisher', 'wrn28_10', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1, 1.0, True,
     '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1, 0.0,
     True, '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False, 0.5, 0.0,
     True, '', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'SGD', 'ElementWiseL2NormAsam', '2', False, False, False, 0.5, 0.0, True,
     '', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 2.0, 0.0,
     True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 3.0, 0.0,
     True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, 0.5, 0.01, False, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_original_etanonzeroCifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, 0.5, 0.01, False, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False,
     0.005, 0.0, True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False,
     0.025, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.025, 0.0,
     True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.25, 0.0,
     True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'Fisher', 'wrn28_10', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1, 1.0, True,
     '--autoaugment', ''],
    ['CIFAR10', 'Fisher', 'wrn28_10', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1, 1.0, True,
     '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1, 0.0,
     True, '--autoaugment', ''],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False, 0.5, 0.0,
     True, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'wrn28_10', 'SGD', 'ElementWiseL2NormAsam', '2', False, False, False, 0.5, 0.0, True,
     '--autoaugment', ''],

    # CIFAR100 RHO fixed
    ['CIFAR100', 'densenet100_nosequential', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True,
     False, False, 1.0, 0.0, True, '', ''],
    ['CIFAR100', 'densenet100_nosequential', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True,
     False, False, 2.5, 0.0, True, '', '--only_bn'],
    ['CIFAR100', 'ASAM_original_etanonzero', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True,
     False, False, 1.0, 0.01, False, '', ''],
    ['CIFAR100', 'ASAM_original_etanonzero', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True,
     False, False, 1.0, 0.01, False, '', '--only_bn'],
    ['CIFAR100', 'densenet100_nosequential', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity',
     True, False, False, 0.01, 0.0, True, '', ''],
    ['CIFAR100', 'densenet100_nosequential', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity',
     True, False, False, 0.05, 0.0, True, '', '--only_bn'],
    ['CIFAR100', 'densenet100_nosequential', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False,
     True, False, 0.05, 0.0, True, '', ''],
    ['CIFAR100', 'densenet100_nosequential', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False,
     True, False, 0.2, 0.0, True, '', '--only_bn'],
    ['CIFAR100', 'Fisher', 'densenet100_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.1, 1.0, True, '', ''],
    ['CIFAR100', 'Fisher', 'densenet100_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.1, 1.0, True, '', '--only_bn'],
    ['CIFAR100', 'densenet100_nosequential', 'densenet100_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False,
     False, False, 0.1, 0.0, True, '', ''],
    ['CIFAR100', 'densenet100_nosequential', 'densenet100_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False,
     False, False, 1.0, 0.0, True, '', '--only_bn'],
    ['CIFAR100', 'densenet100_nosequential', 'densenet100_nosequential', 'SGD', 'ElementWiseL2NormAsam', '2', False,
     False, False, 0.5, 0.0, True, '', ''],
    ['CIFAR100', 'densenet100_nosequential_aa', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 1.0, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'densenet100_nosequential_aa', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 2.5, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'ASAM_original_etanonzero', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True,
     False, False, 1.0, 0.01, False, '--autoaugment', ''],
    ['CIFAR100', 'ASAM_original_etanonzero', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True,
     False, False, 1.0, 0.01, False, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'densenet100_nosequential_aa', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam',
     'infinity', True, False, False, 0.01, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'densenet100_nosequential_aa', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam',
     'infinity', True, False, False, 0.05, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'densenet100_nosequential_aa', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     False, True, False, 0.05, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'densenet100_nosequential_aa', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     False, True, False, 0.2, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'Fisher', 'densenet100_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.1, 1.0, True, '--autoaugment', ''],
    ['CIFAR100', 'Fisher', 'densenet100_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.1, 1.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'densenet100_nosequential_aa', 'densenet100_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2',
     False, False, False, 0.1, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'densenet100_nosequential_aa', 'densenet100_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2',
     False, False, False, 1.0, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'densenet100_nosequential_aa', 'densenet100_nosequential', 'SGD', 'ElementWiseL2NormAsam', '2', False,
     False, False, 0.5, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'resnet56_nosequential', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, 1.0, 0.0, True, '', ''],
    ['CIFAR100', 'resnet56_nosequential', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, 2.5, 0.0, True, '', '--only_bn'],
    ['CIFAR100', 'ASAM_original_etanonzero', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True,
     False, False, 1.0, 0.01, False, '', ''],
    ['CIFAR100', 'ASAM_original_etanonzero', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True,
     False, False, 1.0, 0.01, False, '', '--only_bn'],
    ['CIFAR100', 'resnet56_nosequential', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True,
     False, False, 0.01, 0.0, True, '', ''],
    ['CIFAR100', 'resnet56_nosequential', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True,
     False, False, 0.05, 0.0, True, '', '--only_bn'],
    ['CIFAR100', 'resnet56_nosequential', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True,
     False, 0.05, 0.0, True, '', ''],
    ['CIFAR100', 'resnet56_nosequential', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True,
     False, 0.2, 0.0, True, '', '--only_bn'],
    ['CIFAR100', 'Fisher', 'resnet56_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.1, 1.0, True, '', ''],
    ['CIFAR100', 'Fisher', 'resnet56_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.1, 1.0, True, '', '--only_bn'],
    ['CIFAR100', 'resnet56_nosequential', 'resnet56_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.1, 0.0, True, '', ''],
    ['CIFAR100', 'resnet56_nosequential', 'resnet56_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False,
     False, 1.0, 0.0, True, '', '--only_bn'],
    ['CIFAR100', 'resnet56_nosequential', 'resnet56_nosequential', 'SGD', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.5, 0.0, True, '', ''],
    ['CIFAR100', 'resnet56_nosequential_aa', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True,
     False, False, 1.0, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'resnet56_nosequential_aa', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True,
     False, False, 2.5, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'ASAM_original_etanonzero', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True,
     False, False, 1.0, 0.01, False, '--autoaugment', ''],
    ['CIFAR100', 'ASAM_original_etanonzero', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True,
     False, False, 1.0, 0.01, False, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'resnet56_nosequential_aa', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity',
     True, False, False, 0.01, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'resnet56_nosequential_aa', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity',
     True, False, False, 0.05, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'resnet56_nosequential_aa', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False,
     True, False, 0.05, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'resnet56_nosequential_aa', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False,
     True, False, 0.2, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'Fisher', 'resnet56_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.1, 1.0, True, '--autoaugment', ''],
    ['CIFAR100', 'Fisher', 'resnet56_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
     0.1, 1.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'resnet56_nosequential_aa', 'resnet56_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False,
     False, False, 0.1, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'resnet56_nosequential_aa', 'resnet56_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False,
     False, False, 1.0, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'resnet56_nosequential_aa', 'resnet56_nosequential', 'SGD', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.5, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'resnext_nosequential', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True,
     False, False, 1.0, 0.0, True, '', ''],
    ['CIFAR100', 'resnext_nosequential', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True,
     False, False, 2.5, 0.0, True, '', '--only_bn'],
    ['CIFAR100', 'ASAM_original_etanonzero', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 1.0, 0.01, False, '', ''],
    ['CIFAR100', 'ASAM_original_etanonzero', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 1.0, 0.01, False, '', '--only_bn'],
    ['CIFAR100', 'resnext_nosequential', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity',
     True, False, False, 0.01, 0.0, True, '', ''],
    ['CIFAR100', 'resnext_nosequential', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity',
     True, False, False, 0.05, 0.0, True, '', '--only_bn'],
    ['CIFAR100', 'resnext_nosequential', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False,
     True, False, 0.05, 0.0, True, '', ''],
    ['CIFAR100', 'resnext_nosequential', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False,
     True, False, 0.2, 0.0, True, '', '--only_bn'],
    ['CIFAR100', 'Fisher', 'resnext29_32x4d_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.1, 1.0, True, '', ''],
    ['CIFAR100', 'Fisher', 'resnext29_32x4d_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.1, 1.0, True, '', '--only_bn'],
    ['CIFAR100', 'resnext_nosequential', 'resnext29_32x4d_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False,
     False, False, 0.1, 0.0, True, '', ''],
    ['CIFAR100', 'resnext_nosequential', 'resnext29_32x4d_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False,
     False, False, 1.0, 0.0, True, '', '--only_bn'],
    ['CIFAR100', 'resnext_nosequential', 'resnext29_32x4d_nosequential', 'SGD', 'ElementWiseL2NormAsam', '2', False,
     False, False, 0.5, 0.0, True, '', ''],
    ['CIFAR100', 'resnext_nosequential_aa', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 1.0, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'resnext_nosequential_aa', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 2.5, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'ASAM_original_etanonzero', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 1.0, 0.01, False, '--autoaugment', ''],
    ['CIFAR100', 'ASAM_original_etanonzero', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     True, False, False, 1.0, 0.01, False, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'resnext_nosequential_aa', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam',
     'infinity', True, False, False, 0.01, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'resnext_nosequential_aa', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam',
     'infinity', True, False, False, 0.05, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'resnext_nosequential_aa', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     False, True, False, 0.05, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'resnext_nosequential_aa', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
     False, True, False, 0.2, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'Fisher', 'resnext29_32x4d_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.1, 1.0, True, '--autoaugment', ''],
    ['CIFAR100', 'Fisher', 'resnext29_32x4d_nosequential', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False,
     False, 0.1, 1.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'resnext_nosequential_aa', 'resnext29_32x4d_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2',
     False, False, False, 0.1, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'resnext_nosequential_aa', 'resnext29_32x4d_nosequential', 'SAM_BN', 'ElementWiseL2NormAsam', '2',
     False, False, False, 1.0, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'resnext_nosequential_aa', 'resnext29_32x4d_nosequential', 'SGD', 'ElementWiseL2NormAsam', '2', False,
     False, False, 0.5, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'ASAM_grid_long', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 1.0, 0.0,
     True, '', ''],
    ['CIFAR100', 'ASAM_grid_long', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 2.5, 0.0,
     True, '', '--only_bn'],
    ['CIFAR100', 'ASAM_original_etanonzero', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False,
     1.0, 0.01, False, '', ''],
    ['CIFAR100', 'ASAM_original_etanonzero', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False,
     1.0, 0.01, False, '', '--only_bn'],
    ['CIFAR100', 'ASAM_grid_long', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False, 0.01,
     0.0, True, '', ''],
    ['CIFAR100', 'ASAM_grid_long', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False, 0.05,
     0.0, True, '', '--only_bn'],
    ['CIFAR100', 'ASAM_grid_long', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.05, 0.0,
     True, '', ''],
    ['CIFAR100', 'ASAM_grid_long', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.2, 0.0,
     True, '', '--only_bn'],
    ['CIFAR100', 'Fisher', 'wrn28_10', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1, 1.0, True,
     '', ''],
    ['CIFAR100', 'Fisher', 'wrn28_10', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1, 1.0, True,
     '', '--only_bn'],
    ['CIFAR100', 'ASAM_grid_long', 'wrn28_10', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1, 0.0,
     True, '', ''],
    ['CIFAR100', 'ASAM_grid_long', 'wrn28_10', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False, 1.0, 0.0,
     True, '', '--only_bn'],
    ['CIFAR100', 'ASAM_grid_long', 'wrn28_10', 'SGD', 'ElementWiseL2NormAsam', '2', False, False, False, 0.5, 0.0, True,
     '', ''],
    ['CIFAR100', 'ASAM_grid_long_aa', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 1.0, 0.0,
     True, '--autoaugment', ''],
    ['CIFAR100', 'ASAM_grid_long_aa', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 2.5, 0.0,
     True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'ASAM_original_etanonzero', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False,
     1.0, 0.01, False, '--autoaugment', ''],
    ['CIFAR100', 'ASAM_original_etanonzero', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False,
     1.0, 0.01, False, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'ASAM_grid_long_aa', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False,
     0.01, 0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'ASAM_grid_long_aa', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False,
     0.05, 0.0, True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'ASAM_grid_long_aa', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.05,
     0.0, True, '--autoaugment', ''],
    ['CIFAR100', 'ASAM_grid_long_aa', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.2, 0.0,
     True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'Fisher', 'wrn28_10', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1, 1.0, True,
     '--autoaugment', ''],
    ['CIFAR100', 'Fisher', 'wrn28_10', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1, 1.0, True,
     '--autoaugment', '--only_bn'],
    ['CIFAR100', 'ASAM_grid_long_aa', 'wrn28_10', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False, 0.1, 0.0,
     True, '--autoaugment', ''],
    ['CIFAR100', 'ASAM_grid_long_aa', 'wrn28_10', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False, 1.0, 0.0,
     True, '--autoaugment', '--only_bn'],
    ['CIFAR100', 'ASAM_grid_long_aa', 'wrn28_10', 'SGD', 'ElementWiseL2NormAsam', '2', False, False, False, 0.5, 0.0,
     True, '--autoaugment', '']

]

params_cifar100_etanonzero = [
    # ('ASAM_original_etanonzero', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, ''),  # elementwise l2 ASAM
    # ('ASAM_original_etanonzero', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, '--only_bn'), # ASAM BN
    # ('ASAM_original_etanonzero', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, '--no_bn'), # ASAM NO BN
    ('ASAM_original_etanonzero', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, ''),  # elementwise l2

]
params_cifar100_fillup = [
    ('ASAM_original_etanonzero', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.02,  ''),  # elementwise l2

    # elementwise l2
]
params_cifar10_etanonzero = [
    # ('ASAM_original_etanonzeroCifar10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, ''),  # elementwise l2 ASAM
    # ('ASAM_original_etanonzeroCifar10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, '--only_bn'), # ASAM BN
    ('ASAM_original_etanonzeroCifar10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, '--no_bn'), # ASAM NO BN
    # elementwise l2
]

params_fisher = [
    # ('Fisher', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, '--only_bn'),
    # ('Fisher', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, ''),
    ('Fisher', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False, '--no_bn'),
    # ASAM BN
]

params_cifar10_fillup_=[
    ('ASAM_original_etanonzero', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, ''),  # elementwise l2
    # ('ASAM_original_etanonzeroCifar10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, '--only_bn'),
    # elementwise l2
]

params_track_weights=[
    ('TrackBNWeights_etanonzero', 'SGD', 'L2Norm', '2', False, False, False, 0.5, ''),  # SGD
    ('TrackBNWeights_etanonzero', 'SAM_BN', 'L2Norm', '2', False, False, False, 0.2, ''),  # SAM
    ('TrackBNWeights_etanonzero', 'SAM_BN', 'L2Norm', '2', False, False, False, 1., '--only_bn'),  # SAM
    ('TrackBNWeights_etanonzero', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5., '--only_bn'),  # Elementwise l2
    ('TrackBNWeights_etanonzero', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5., ''),  # Elementwise l2
    ('TrackBNWeights_etanonzero', 'ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.05, '--only_bn'),  # elementwise Linf
    ('TrackBNWeights_etanonzero', 'ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.01, ''),  # elementwise Linf
]

param_plot = [
# ['CIFAR100', 'ASAM_grid_long', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.05, 0.0,
#      True, '', '--no_bn'],
# ['CIFAR100', 'ASAM_grid_long_aa', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 0.05, 0.0,
#      True, '--autoaugment', ''],
#     ['CIFAR100', 'resnet56_nosequential_aa', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False,
#      True, False, 0.01, 0.0, True, '--autoaugment', ''],
#     ['CIFAR100', 'resnet56_nosequential', 'resnet56_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False,
#      True, False, 0.01, 0.0, True, '', ''],
#     ['CIFAR100', 'densenet100_nosequential_aa', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
#      False, True, False, 0.01, 0.0, True, '--autoaugment', ''],
#     ['CIFAR100', 'densenet100_nosequential', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
#      False, True, False, 0.01, 0.0, True, '', ''],
#     ['CIFAR100', 'resnext_nosequential_aa', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
#      False, True, False, 0.01, 0.0, True, '--autoaugment', ''],
#     ['CIFAR100', 'resnext_nosequential', 'resnext29_32x4d_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2',
#      False, True, False, 0.01, 0.0, True, '', ''],
#     ['CIFAR100', 'ASAM_grid_long', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.01, 0.0,
#      True, '', ''],
#     ['CIFAR100', 'ASAM_grid_long_aa', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.01,
#      0.0, True, '--autoaugment', ''],
#     ['CIFAR100', 'Fisher', 'wrn28_10', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
#      0.1, 1.0, True, '', '--no_bn'],
#     ['CIFAR100', 'Fisher', 'wrn28_10', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
#      0.05, 1.0, True, '', '--no_bn'],
#     ['CIFAR100', 'Fisher', 'wrn28_10', 'FISHER_SAM', 'ElementWiseL2NormAsam', '2', False, False, False,
#      0.5, 1.0, True, '', '--no_bn'],
['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, .5, 0.01, False, '--autoaugment', '--only_bn'],
    ['CIFAR10', 'ASAM_BN_Cifar10', 'densenet100_nosequential', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False,
     False, .5, 0.01, False, '--autoaugment', ''],
]
param_antiSAM = [
['CIFAR100', 'Anti_SAM', 'wrn28_10', 'SGD', 'ElementWiseL2NormAsam', '2', False, False, False, 0.5, 0.0, True, '', ''],
['CIFAR100', 'Anti_SAM', 'wrn28_10', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False, 0.2, 0.0, True, '', ''],
['CIFAR100', 'Anti_SAM', 'wrn28_10', 'SAM_BN', 'ElementWiseL2NormAsam', '2', False, False, False, 1., 0.0, True, '', '--only_bn'],
['CIFAR100', 'Anti_SAM', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5., 0.0, True, '', ''],
['CIFAR100', 'Anti_SAM', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 10., 0.0, True, '', '--only_bn'],
['CIFAR100', 'Anti_SAM', 'wrn28_10', 'ASAM_BN', 'ElementWiseL2NormAsam', 'infinity', True, False, False, 0.1, 0.0, True, '', '--only_bn'],
]

param_adamw = [
['CIFAR100', 'AdamW_test', 'vit_t', 'SGD', 'ElementWiseL2NormAsam', '2', False, False, False, 0.5, 0.0, True, '', '', 0.1],
['CIFAR100', 'AdamW_test', 'vit_t', 'AdamW', 'ElementWiseL2NormAsam', '2', False, False, False, 0.5, 0.0, True, '', '', 0.1],
['CIFAR100', 'AdamW_test', 'vit_t', 'AdamW', 'ElementWiseL2NormAsam', '2', False, False, False, 0.5, 0.0, True, '', '', 0.01],
['CIFAR100', 'AdamW_test', 'vit_t', 'AdamW', 'ElementWiseL2NormAsam', '2', False, False, False, 0.5, 0.0, True, '', '', 0.001],
['CIFAR100', 'AdamW_test', 'vit_t', 'AdamW',  'ElementWiseL2NormAsam', '2', False, False, False, 0.5, 0.0, True, '', '', 0.0001],
]
params_cifar10_5=[
('SAM_BN', 'L2Norm', '2', False, False, False, 5., '--only_bn'),  # SAM
('SAM_BN', 'L2Norm', '2', False, False, False, 10., '--only_bn'),  # SAM
('SAM_BN', 'L2Norm', '2', False, False, False, 25., '--only_bn'),  # SAM
('SAM_BN', 'L2Norm', '2', False, False, False, 0.25, ''),  # SAM
('SAM_BN', 'L2Norm', '2', False, False, False, 0.025, ''),  # SAM
('SAM_BN', 'L2Norm', '2', False, False, False, 0.01, ''),  # SAM
('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 25., ''),  # Elementwise l2
('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 25., '--only_bn'),  # Elementwise l2
('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 2.5, '--only_bn'),  # Elementwise l2
('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 2.5, ''),  # Elementwise l2
('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, 5., ''),  # Elementwise l2
('ASAM_BN', 'ElementWiseL2NormAsam', '2', True, False, False, .5, ''),  # Elementwise l2
('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.25, '--only_bn'),  # layerwise l2
('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.5, '--only_bn'),  # layerwise l2
('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 2.5, '--only_bn'),  # layerwise l2
('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.0025, ''),  # layerwise l2
('ASAM_BN', 'ElementWiseL2NormAsam', '2', False, True, False, 0.001, ''),  # layerwise l2
('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.0025, ''), # elementwise Linf
('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.0005, ''), # elementwise Linf
('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.25, '--only_bn'), # elementwise Linf
('ASAM_BN', 'ElementwiseLinfNormAsam', 'infinity', True, False, False, 0.5, '--only_bn'), # elementwise Linf
('SGD', 'ElementwiseLinfNormAsam', 'infinity', False, False, False, 0.5, ''), # SGD
('AdamW', 'ElementwiseLinfNormAsam', 'infinity', False, False, False, 0.5, ''), # AdamW
]
params_cifar10_adamw=[
         ('AdamW', 'L2Norm', '2', False, False, False, 0.5, ''),  # AdamW
]

params_cifar10_start_end_sam=[
         ('SAM_BN', 'L2Norm', '2', False, False, False, 0.25, ''),  # SAM
         ('SAM_BN', 'L2Norm', '2', False, False, False, 5., '--only_bn'),  # SAM
]

base_rhos = [1., 2.5, 5., 10.]
rhos = [rho*(10*(exp)) for exp in [0.00001,0.0001,0.001,0.01, 0.1, 1.] for rho in base_rhos]
import time
 # 'ASAM_BN_Cifar10' # resnext_nosequential
for seed in [0]:
    # 'PreActResNet34BatchNorm', 'PreActResNet34LayerNorm' 'PreActResNet34GroupNorm' PreActResNet34LayerNormSmall
    # ConvMixerGroup, ConvMixerBatch, ConvMixerLayer
    # for model in ['wrn28_10', 'resnext29_32x4d_nosequential', 'resnet56_nosequential', 'densenet100_nosequential']: #['PreActResNet34LayerNorm', 'PreActResNet34GroupNorm', 'PreActResNet34LayerNormSmall']: #resnext29_32x4d_nosequential #resnet56_nosequential #densenet100_nosequential
    #     for dataset in ['CIFAR100']:
            # model = 'densenet100_nosequential' #
    no_grad_norm=''
    lr=0.0001
    momentum=0.9
    weight_decay=5e-4
    epochs=200
    smoothing=0.1
    cutmix='' # --cutmix
    eta = 0.0 #0.01
    batch_size=64
    # normalize_bias='--normalize_bias' #'--normalize_bias'
    # autoaugment = ''
    eval_ascent = ''
    ascent_with_old_batch=''
    no_bn=''
    random_idxs_frac=0.
    # for only_bn, no_bn in [
    #     ('--only_bn', ''),
    #     # ('', '--no_bn'),
    #     ('','')
    # ]:
    # for rho in [0.05, 1., 0.5]:
    # for autoaugment in ['', '--autoaugment']:
    # dataset='CIFAR100'
    autoaugment='--autoaugment'
    for dataset in ['CIFAR10','CIFAR100']:
        for model in ['vit_t','vit_s']:
            for minimizer, norm_adaptive, p, e, l, f, rho, only_bn in params_cifar10+params_cifar10_2+params_cifar10_3+params_cifar10_4+params_cifar10_5:#params+params_2+params_3+params_4+params_5+params_8+params_9+params_10: #params+params_2+params_3+params_4+params_5+params_8+params_9:
            #         for folder, minimizer, norm_adaptive, p, e, l, f, rho, only_bn in params_cifar100_fillup:  # params+params_2+params_3+params_4+params_5+params_8+params_9+params_10: #params+params_2+params_3+params_4+params_5+params_8+params_9:
                if model=='vit_s' and dataset=='CIFAR10':
                    folder='vits_cifar_10classes'
                else:
                    folder='vits_cifar'
                # folder='shift_m'
                for m in [batch_size]:#[int(batch_size/i) for i in [1,2,4,8]]: #
                    # for start_sam, end_sam in [(0,50),
                    #                            (0,100),
                    #                            (0,150),
                    #                            (0,200),
                    #                            (50,1000),
                    #                            (100,1000),
                    #                            (150,1000),
                    #                            (200,1000),
                    #                            ]:
                    start_sam = 0
                    end_sam=10000
                    if rho not in rhos:
                            continue
                    # if minimizer not in ['SAM_BN', 'AdamW', 'SGD']:
                    #     continue
                    # include no-bn
                    if only_bn=='--only_bn':
                         continue
                    else:
                         only_bn='--no_bn'
                    elementwise = '--elementwise' if e else ''
                    layerwise = '--layerwise' if l else ''
                    filterwise = '--filterwise' if f else ''
                    normalize_bias='--normalize_bias' #if n else '' #'--normalize_bias'
                    save = './snapshots/' + folder
                    data_path = '/mnt/qb/hein/datasets/'+dataset
                    base_optimizer='--base_minimizer SGD' if minimizer=='SGD' else '--base_minimizer AdamW'
                    with open(loop, 'w+') as fh:
                        fh.writelines('#!/bin/bash\n')
                        fh.writelines(
                            '#SBATCH --ntasks=1                # Number of tasks (see below)\n')
                        fh.writelines(
                            '#SBATCH --cpus-per-task=4       # Number of CPU cores per task\n')
                        fh.writelines(
                            '#SBATCH --nodes=1                 # Ensure that all cores are on one machine\n')
                        fh.writelines('#SBATCH --time=1-23:00            # Runtime in D-HH:MM\n')
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
                            'python ExtraASAM.py --dataset {} --data_path {} --model {} --minimizer {} --lr {} --momentum {} --weight_decay {} --batch_size {} --epochs {} --smoothing {} --rho {} --p {} {} {} {} {} {} --eta {} --m {} --save {} --norm_adaptive {} {} --seed {} --random_idxs_frac {} --start_sam {} --end_sam {} {} {} {} {} {} {}'.format(dataset, data_path, model, minimizer, lr, momentum, weight_decay, batch_size, epochs, smoothing, rho, p, layerwise, filterwise, elementwise, autoaugment, cutmix, eta, m, save, norm_adaptive, normalize_bias, seed, random_idxs_frac, start_sam, end_sam, eval_ascent, ascent_with_old_batch, only_bn, no_bn, no_grad_norm, base_optimizer))
                    # os.system('sbatch -p gpu-2080ti %s' % loop)
                    os.system('sbatch -p gpu-2080ti %s' % loop)
                    time.sleep(2)  # Sleep for 2 seconds in order to minimize race conflicts for folder creation