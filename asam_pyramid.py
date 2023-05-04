import torch.nn as nn
import torch
from collections import defaultdict
from collections import defaultdict
from torch.optim import Optimizer
import numpy
class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01, adaptive=True, p='2', normalize_bias=False, elementwise=True, filterwise=False, layerwise=False):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
        self.adaptive=adaptive
        self.p=p
        self.normalize_bias=normalize_bias
        self.elementwise=elementwise
        self.filterwise=filterwise
        self.layerwise=layerwise


    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            self.state[p]['old_p'] = p.data.clone()
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None: # initialize t_w
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if ('weight' in n) or self.normalize_bias: # compute t_w and modify grad to t_w*grad
                t_w[...] = p[...]
                if self.elementwise:
                    t_w.abs_().add_(self.eta)
                elif self.layerwise:
                    t_w.data = torch.norm(p.data)*torch.ones_like(p.data).add_(self.eta).data
                elif self.filterwise:
                    if 'conv' in n:
                        norms = torch.linalg.norm(p.data, dim=[1, 2, 3])  # compute norms
                        norms = norms.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
                            [1] + list(p[0, ...].shape))  # bring into shape of param
                        t_w.data = norms
                    else:
                        t_w.abs_().add_(self.eta)
                if self.p == '2':
                    p.grad.mul_(t_w)  # update gradient
            if self.p == 'infinity':
                if ('weight' in n) or self.normalize_bias:
                    p.grad.sign_().mul_(t_w)
                else:
                    p.grad.sign_()
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps") # get normalization operator
            if self.p=='2':
                if ('weight' in n) or self.normalize_bias: # second multiplication with t_w
                    p.grad.mul_(t_w)
                eps = t_w
                eps[...] = p.grad[...]
                eps.mul_(self.rho / wgrad_norm)
            elif self.p=='infinity':
                eps = t_w
                eps[...] = p.grad[...]*self.rho
            else:
                raise NotImplementedError
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.data = self.state[p]['old_p']

        self.optimizer.step()
        self.optimizer.zero_grad()


    @torch.no_grad()
    def accumulate_grad_and_resume(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.data = self.state[p]['old_p'] # resume old max
            grad_accum = self.state[p].get('grad_accum')
            if grad_accum is None:
                self.state[p]['grad_accum']=p.grad.detach().clone()
            else:
                self.state[p]['grad_accum']=grad_accum+p.grad.detach().clone()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_with_accumulated_grad(self,steps):
        for i, (n, p) in enumerate(self.model.named_parameters()):
            if p.grad is None:
                continue
            p.data = self.state[p]['old_p'] # resume old max
            grad_accum = self.state[p].get('grad_accum')
            if grad_accum is None:
                raise ValueError('No Gradient accumulated for {}!'.format(n))
            else:
                p.grad.data=grad_accum/steps
            self.state[p]['grad_accum']=None
        self.optimizer.step()
        self.optimizer.zero_grad()



class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            self.state[p]['old_p'] = p.data.clone()
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()


class ExtraASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01, adaptive=True, p='2', normalize_bias=False,
                 elementwise=True, filterwise=False, layerwise=False):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
        self.adaptive = adaptive
        self.p = p
        self.normalize_bias = normalize_bias
        self.elementwise = elementwise
        self.filterwise = filterwise
        self.layerwise = layerwise

        # initialize old_p for first step
        for n, p in self.model.named_parameters():
            self.state[p]['old_p'] = p.data.clone()

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if self.state[p]['old_grad'] is None:
                print('none')
                continue
            p.grad.data.zero_().add_(self.state[p]['old_grad'])
            self.state[p]['old_p'] = p.data.clone()
            t_w = self.state[p].get("eps")
            if t_w is None:  # initialize t_w
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if ('weight' in n) or self.normalize_bias:  # compute t_w and modify grad to t_w*grad
                t_w[...] = p[...]
                if self.elementwise:
                    t_w.abs_().add_(self.eta)
                elif self.layerwise:
                    t_w.data = torch.norm(p.data) * torch.ones_like(p.data).add_(self.eta).data
                elif self.filterwise:
                    if 'conv' in n:
                        norms = torch.linalg.norm(p.data, dim=[1, 2, 3])  # compute norms
                        norms = norms.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
                            [1] + list(p[0, ...].shape))  # bring into shape of param
                        t_w.data = norms
                    else:
                        t_w.abs_().add_(self.eta)
                if self.p == '2':
                    p.grad.mul_(t_w)  # update gradient
            if self.p == 'infinity':
                if ('weight' in n) or self.normalize_bias:
                    p.grad.sign_().mul_(t_w)
                else:
                    p.grad.sign_()
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")  # get normalization operator
            if self.p == '2':
                if ('weight' in n) or self.normalize_bias:  # second multiplication with t_w
                    p.grad.mul_(t_w)
                eps = t_w
                eps[...] = p.grad[...]
                eps.mul_(self.rho / wgrad_norm)
            elif self.p == 'infinity':
                eps = t_w
                eps[...] = p.grad[...] * self.rho
            else:
                raise NotImplementedError
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                self.state[p]['old_grad'] = None
                continue
            self.state[p]['old_grad'] = p.grad.data.clone()#.detach()
            p.data = self.state[p]['old_p']

        self.optimizer.step()
        self.optimizer.zero_grad()


class ExtraSAM(ExtraASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            self.state[p]['old_p'] = p.data.clone()
            if self.state[p]['old_grad'] is None:
                continue
            p.grad.data = self.state[p]['old_grad'].clone().detach()
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

class SAM_mock:
    '''
    mocks the reformulated SAM algorithm by artificially generating adversarial steps
    '''
    def __init__(self, optimizer, model, rho=0.5, alpha=0.):
        self.optimizer = optimizer
        self.model = model
        self.state = defaultdict(dict)
        self.rho = rho
        self.alpha = alpha

        # initialize
        for n, p in self.model.named_parameters():
            self.state[p]['grad_perturbed_old'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self):
        # compute perturbed gradient and its norm
        norms = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                self.state[p]['grad_perturbed'] = None
                continue
            g = p.grad.data.clone().detach()
            grad_perturbed = g*(1+self.alpha*2*(torch.rand_like(g)-0.5))
            self.state[p]['grad_perturbed'] = grad_perturbed
            norms.append(torch.norm(grad_perturbed, p=2))
        norm = torch.norm(torch.stack(norms), p=2)

        # perturb, i.e. add diff of normalized gradients to weight
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            self.state[p]['grad_perturbed'] = self.state[p]['grad_perturbed'].clone() / norm
            p.add_(self.rho*(self.state[p]['grad_perturbed']-self.state[p]['grad_perturbed_old']))
            self.state[p]['grad_perturbed_old'] = self.state[p]['grad_perturbed'].clone().detach()

        # do conventional gradient step
        self.optimizer.step()
        self.optimizer.zero_grad()


class ASAM_BN:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01, adaptive=True, p='2', normalize_bias=False, elementwise=True, filterwise=False, layerwise=False, no_bn=False, only_bn=False, update_grad=False, no_grad_norm=False):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
        self.adaptive=adaptive
        self.p=p
        self.normalize_bias=normalize_bias
        self.elementwise=elementwise
        self.filterwise=filterwise
        self.layerwise=layerwise
        self.only_bn=only_bn
        self.no_bn=no_bn
        self.update_grad=update_grad
        self.no_grad_norm = no_grad_norm
        assert not (self.only_bn and self.no_bn)

    @torch.no_grad()
    def ascent_step(self):
        if self.update_grad:
            update_grad(self.optimizer)
        wgrads = []
        for n, p in self.model.named_parameters():
            self.state[p]['old_p'] = p.data.clone()
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            t_w = self.state[p].get("eps")
            if t_w is None: # initialize t_w
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if ('weight' in n) or self.normalize_bias: # compute t_w and modify grad to t_w*grad
                t_w[...] = p[...]
                if self.elementwise:
                    t_w.abs_().add_(self.eta)
                elif self.layerwise:
                    t_w.data = torch.norm(p.data)*torch.ones_like(p.data).add_(self.eta).data
                elif self.filterwise:
                    if 'conv' in n:
                        norms = torch.linalg.norm(p.data, dim=[1, 2, 3])  # compute norms
                        norms = norms.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
                            [1] + list(p[0, ...].shape))  # bring into shape of param
                        t_w.data = norms
                    else:
                        t_w.abs_().add_(self.eta)
                if self.p == '2':
                    p.grad.mul_(t_w)  # update gradient
            if self.p == 'infinity':
                if ('weight' in n) or self.normalize_bias:
                    p.grad.sign_().mul_(t_w)
                else:
                    p.grad.sign_()
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16

        for n, p in self.model.named_parameters():
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            t_w = self.state[p].get("eps") # get normalization operator
            if self.p=='2':
                if ('weight' in n) or self.normalize_bias: # second multiplication with t_w
                    p.grad.mul_(t_w)
                eps = t_w
                eps[...] = p.grad[...]
                eps.mul_(self.rho / wgrad_norm)
            elif self.p=='infinity':
                eps = t_w
                eps[...] = p.grad[...]*self.rho
            else:
                raise NotImplementedError
            # p.sub_(eps) # Anti-SAM
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            p.data = self.state[p]['old_p']

        self.optimizer.step()
        self.optimizer.zero_grad()

class SAM_BN(ASAM_BN):
    @torch.no_grad()
    def ascent_step(self):
        if self.update_grad:
            if 'Shampoo' in self.optimizer.__class__.__name__:
                precondition_gradient_with_shampoo(self.optimizer)
            else:
                update_grad(self.optimizer)
        grads = []
        for n, p in self.model.named_parameters():
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            self.state[p]['old_p'] = p.data.clone()
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = 1. if self.no_grad_norm else torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps) # Anti-SAM
            # p.sub_(eps)
        self.optimizer.zero_grad()



class FISHER_SAM:
    def __init__(self, optimizer, model, rho=0.1, eta=0.01, adaptive=True, p='2', normalize_bias=True, elementwise=True, filterwise=False, layerwise=False, no_bn=False, only_bn=False, update_grad=False, no_grad_norm=False):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
        self.adaptive=adaptive
        self.p=p
        self.normalize_bias=normalize_bias
        self.elementwise=elementwise
        self.filterwise=filterwise
        self.layerwise=layerwise
        self.only_bn=only_bn
        self.no_bn=no_bn
        self.update_grad=update_grad
        self.no_grad_norm = no_grad_norm
        assert not (self.only_bn and self.no_bn)

    @torch.no_grad()
    def ascent_step(self):
        if self.update_grad:
            update_grad(self.optimizer)
        wgrads = []
        for n, p in self.model.named_parameters():
            self.state[p]['old_p'] = p.data.clone()
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            t_w = self.state[p].get("eps")
            if t_w is None: # initialize t_w
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if ('weight' in n) or self.normalize_bias: # compute t_w and modify grad to t_w*grad
                t_w[...] = torch.sqrt(1/(1 + self.eta*p.grad[...]**2)) # t_w = F^(-0.5)
                if self.p == '2':
                    p.grad.mul_(t_w)  # update gradient
            if self.p == 'infinity':
                if ('weight' in n) or self.normalize_bias:
                    p.grad.sign_().mul_(t_w)
                else:
                    p.grad.sign_()
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16

        for n, p in self.model.named_parameters():
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            t_w = self.state[p].get("eps") # get normalization operator
            if self.p=='2':
                if ('weight' in n) or self.normalize_bias: # second multiplication with t_w
                    p.grad.mul_(t_w)
                eps = t_w
                eps[...] = p.grad[...]
                eps.mul_(self.rho / wgrad_norm)
            elif self.p=='infinity':
                eps = t_w
                eps[...] = p.grad[...]*self.rho
            else:
                raise NotImplementedError
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            p.data = self.state[p]['old_p']

        self.optimizer.step()
        self.optimizer.zero_grad()


class ASAM_2RHO:
    def __init__(self, optimizer, model, rho_conv=0.5, rho_bn=0.5, eta=0.01, adaptive=True, p='2', normalize_bias=False, elementwise=True, filterwise=False, layerwise=False, no_bn=False, only_bn=False, update_grad=False, no_grad_norm=False):
        self.optimizer = optimizer
        self.model = model
        self.rho_conv = rho_conv
        self.rho_bn = rho_bn
        self.eta = eta
        self.state = defaultdict(dict)
        self.adaptive=adaptive
        self.p=p
        self.normalize_bias=normalize_bias
        self.elementwise=elementwise
        self.filterwise=filterwise
        self.layerwise=layerwise
        self.only_bn=only_bn
        self.no_bn=no_bn
        self.update_grad=update_grad
        self.no_grad_norm = no_grad_norm
        assert not (self.only_bn and self.no_bn)

    @torch.no_grad()
    def ascent_step(self):
        if self.update_grad:
            update_grad(self.optimizer)
        wgrads_bn = []
        wgrads_nobn = []
        for n, p in self.model.named_parameters():
            self.state[p]['old_p'] = p.data.clone()
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            t_w = self.state[p].get("eps")
            if t_w is None: # initialize t_w
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if ('weight' in n) or self.normalize_bias: # compute t_w and modify grad to t_w*grad
                t_w[...] = p[...]
                if self.elementwise:
                    t_w.abs_().add_(self.eta)
                elif self.layerwise:
                    t_w.data = torch.norm(p.data)*torch.ones_like(p.data).add_(self.eta).data
                elif self.filterwise:
                    if 'conv' in n:
                        norms = torch.linalg.norm(p.data, dim=[1, 2, 3])  # compute norms
                        norms = norms.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
                            [1] + list(p[0, ...].shape))  # bring into shape of param
                        t_w.data = norms
                    else:
                        t_w.abs_().add_(self.eta)
                if self.p == '2':
                    p.grad.mul_(t_w)  # update gradient
            if self.p == 'infinity':
                if ('weight' in n) or self.normalize_bias:
                    p.grad.sign_().mul_(t_w)
                else:
                    p.grad.sign_()
            if ('norm' in n or 'bn' in n):
                wgrads_bn.append(torch.norm(p.grad, p=2))
            else:
                wgrads_nobn.append(torch.norm(p.grad, p=2))
        wgrad_norm_bn = torch.norm(torch.stack(wgrads_bn), p=2) + 1.e-16
        wgrad_norm_nobn = torch.norm(torch.stack(wgrads_nobn), p=2) + 1.e-16

        for n, p in self.model.named_parameters():
            rho = self.rho_bn if ('norm' in n or 'bn' in n) else self.rho_conv
            wgrad_norm = wgrad_norm_bn if ('norm' in n or 'bn' in n) else wgrad_norm_nobn
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            t_w = self.state[p].get("eps") # get normalization operator
            if self.p=='2':
                if ('weight' in n) or self.normalize_bias: # second multiplication with t_w
                    p.grad.mul_(t_w)
                eps = t_w
                eps[...] = p.grad[...]
                eps.mul_(rho / wgrad_norm)
            elif self.p=='infinity':
                eps = t_w
                eps[...] = p.grad[...]*rho
            else:
                raise NotImplementedError
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            p.data = self.state[p]['old_p']

        self.optimizer.step()
        self.optimizer.zero_grad()


class SAM_2RHO(ASAM_2RHO):
    @torch.no_grad()
    def ascent_step(self):
        grads_bn = []
        grads_nobn = []
        for n, p in self.model.named_parameters():
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            self.state[p]['old_p'] = p.data.clone()
            if ('norm' in n or 'bn' in n):
                grads_bn.append(torch.norm(p.grad, p=2))
            else:
                grads_nobn.append(torch.norm(p.grad, p=2))
        grad_norm_bn = torch.norm(torch.stack(grads_bn), p=2) + 1.e-16
        grad_norm_nobn = torch.norm(torch.stack(grads_nobn), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            rho = self.rho_bn if ('norm' in n or 'bn' in n) else self.rho_conv
            grad_norm = grad_norm_bn if ('norm' in n or 'bn' in n) else grad_norm_nobn
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()




class ASAM_BN_WEIGHTS:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01, adaptive=True, p='2', normalize_bias=False, elementwise=True, filterwise=False, layerwise=False, no_bn=False, only_bn=False, update_grad=False, no_grad_norm=False):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
        self.adaptive=adaptive
        self.p=p
        self.normalize_bias=normalize_bias
        self.elementwise=elementwise
        self.filterwise=filterwise
        self.layerwise=layerwise
        self.only_bn=only_bn
        self.no_bn=no_bn
        self.update_grad=update_grad
        self.no_grad_norm = no_grad_norm
        self.name_to_optim='weight'
        assert not (self.only_bn and self.no_bn)

    @torch.no_grad()
    def ascent_step(self):
        if self.update_grad:
            update_grad(self.optimizer)
        wgrads = []
        for n, p in self.model.named_parameters():
            self.state[p]['old_p'] = p.data.clone()
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n) or self.name_to_optim not in n:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None: # initialize t_w
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if ('weight' in n) or self.normalize_bias: # compute t_w and modify grad to t_w*grad
                t_w[...] = p[...]
                if self.elementwise:
                    t_w.abs_().add_(self.eta)
                elif self.layerwise:
                    t_w.data = torch.norm(p.data)*torch.ones_like(p.data).add_(self.eta).data
                elif self.filterwise:
                    if 'conv' in n:
                        norms = torch.linalg.norm(p.data, dim=[1, 2, 3])  # compute norms
                        norms = norms.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
                            [1] + list(p[0, ...].shape))  # bring into shape of param
                        t_w.data = norms
                    else:
                        t_w.abs_().add_(self.eta)
                if self.p == '2':
                    p.grad.mul_(t_w)  # update gradient
            if self.p == 'infinity':
                if ('weight' in n) or self.normalize_bias:
                    p.grad.sign_().mul_(t_w)
                else:
                    p.grad.sign_()
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16

        for n, p in self.model.named_parameters():
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n) or self.name_to_optim not in n:
                continue
            t_w = self.state[p].get("eps") # get normalization operator
            if self.p=='2':
                if ('weight' in n) or self.normalize_bias: # second multiplication with t_w
                    p.grad.mul_(t_w)
                eps = t_w
                eps[...] = p.grad[...]
                eps.mul_(self.rho / wgrad_norm)
            elif self.p=='infinity':
                eps = t_w
                eps[...] = p.grad[...]*self.rho
            else:
                raise NotImplementedError
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n) or self.name_to_optim not in n:
                continue
            p.data = self.state[p]['old_p']

        self.optimizer.step()
        self.optimizer.zero_grad()

class SAM_BN_WEIGHTS(ASAM_BN_WEIGHTS):
    @torch.no_grad()
    def ascent_step(self):
        if self.update_grad:
            if 'Shampoo' in self.optimizer.__class__.__name__:
                precondition_gradient_with_shampoo(self.optimizer)
            else:
                update_grad(self.optimizer)
        grads = []
        for n, p in self.model.named_parameters():
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n) or self.name_to_optim not in n:
                continue
            self.state[p]['old_p'] = p.data.clone()
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = 1. if self.no_grad_norm else torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n) or self.name_to_optim not in n:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

class ASAM_BN_FC(ASAM_BN_WEIGHTS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name_to_optim='fc'

class SAM_BN_FC(SAM_BN_WEIGHTS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name_to_optim='fc'

class AVG_ASAM_BN:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01, adaptive=True, p='2', normalize_bias=False, elementwise=True, filterwise=False, layerwise=False, no_bn=False, only_bn=False, update_grad=False, no_grad_norm=False):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
        self.adaptive=adaptive
        self.p=p
        self.normalize_bias=normalize_bias
        self.elementwise=elementwise
        self.filterwise=filterwise
        self.layerwise=layerwise
        self.only_bn=only_bn
        self.no_bn=no_bn
        self.update_grad=update_grad
        self.no_grad_norm = no_grad_norm
        assert not (self.only_bn and self.no_bn)

    @torch.no_grad()
    def ascent_step(self):
        if self.update_grad:
            update_grad(self.optimizer)
        wgrads = []
        for n, p in self.model.named_parameters():
            self.state[p]['old_p'] = p.data.clone()
            if (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            p.grad = (torch.randn_like(p.data)) # random perturbation
            t_w = self.state[p].get("eps")
            if t_w is None: # initialize t_w
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if ('weight' in n) or self.normalize_bias: # compute t_w and modify grad to t_w*grad
                t_w[...] = p[...]
                if self.elementwise:
                    t_w.abs_().add_(self.eta)
                elif self.layerwise:
                    t_w.data = torch.norm(p.data)*torch.ones_like(p.data).add_(self.eta).data
                elif self.filterwise:
                    if 'conv' in n:
                        norms = torch.linalg.norm(p.data, dim=[1, 2, 3])  # compute norms
                        norms = norms.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
                            [1] + list(p[0, ...].shape))  # bring into shape of param
                        t_w.data = norms
                    else:
                        t_w.abs_().add_(self.eta)
                if self.p == '2':
                    p.grad.mul_(t_w)  # update gradient
            if self.p == 'infinity':
                if ('weight' in n) or self.normalize_bias:
                    p.grad.sign_().mul_(t_w)
                else:
                    p.grad.sign_()
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16

        for n, p in self.model.named_parameters():
            if (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            t_w = self.state[p].get("eps") # get normalization operator
            if self.p=='2':
                if ('weight' in n) or self.normalize_bias: # second multiplication with t_w
                    p.grad.mul_(t_w)
                eps = t_w
                eps[...] = p.grad[...]
                eps.mul_(self.rho / wgrad_norm)
            elif self.p=='infinity':
                eps = t_w
                eps[...] = p.grad[...]*self.rho
            else:
                raise NotImplementedError
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            p.data = self.state[p]['old_p']

        self.optimizer.step()
        self.optimizer.zero_grad()

class AVG_SAM_BN(ASAM_BN):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            p.grad = (torch.randn_like(p.data))  # random perturbation
            self.state[p]['old_p'] = p.data.clone()
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = 1. if self.no_grad_norm else torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()
# sampling perturbation once for all layers, then adding it
# class AVG_ASAM_BN:
#     def __init__(self, optimizer, model, rho=0.5, eta=0.01, adaptive=True, p='2', normalize_bias=False, elementwise=True, filterwise=False, layerwise=False, no_bn=False, only_bn=False, update_grad=False, no_grad_norm=False):
#         self.optimizer = optimizer
#         self.model = model
#         self.rho = rho
#         self.eta = eta
#         self.state = defaultdict(dict)
#         self.adaptive=adaptive
#         self.p=p
#         self.normalize_bias=normalize_bias
#         self.elementwise=elementwise
#         self.filterwise=filterwise
#         self.layerwise=layerwise
#         self.only_bn=only_bn
#         self.no_bn=no_bn
#         assert not (self.only_bn and self.no_bn)
#         self.n_params = sum([p.numel() for n,p in self.model.named_parameters() if ((self.no_bn and ('norm' not in n and 'bn' not in n)) or (self.only_bn and ('norm' in n or 'bn' in n)) ) or self.no_bn==self.only_bn==False])
#
#     @torch.no_grad()
#     def uniform_sphere_numpy(self, batch_size, dim, epsilon=1., cuda=False, ord=2):
#         if ord != 2:
#             raise NotImplementedError
#         u = numpy.random.normal(0, 1, batch_size * dim).reshape(batch_size,
#                                                              dim)  # an array of d normally distributed random variables
#         norm = numpy.linalg.norm(u, axis=1).reshape(-1, 1)
#         x = epsilon * u / norm
#         random = torch.from_numpy(x).float()
#         return random.cuda() if cuda else random
#
#     @torch.no_grad()
#     def ascent_step(self):
#         perturbation = self.uniform_sphere_numpy(1, self.n_params, cuda=True)
#         n_i = 0
#         wgrads = []
#         for n, p in self.model.named_parameters():
#             self.state[p]['old_p'] = p.data.clone()
#             if (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
#                 continue
#             size_i = list(p.data.shape)
#             perturbation_layer = perturbation[n_i: n_i + numpy.prod(size_i)].view(size_i)
#             n_i += numpy.prod(size_i)
#             t_w = self.state[p].get("eps")
#             if t_w is None: # initialize t_w
#                 t_w = torch.clone(p).detach()
#                 self.state[p]["eps"] = t_w
#             if ('weight' in n) or self.normalize_bias: # compute t_w and modify grad to t_w*grad
#                 t_w[...] = p[...]
#                 if self.elementwise:
#                     t_w.abs_().add_(self.eta)
#                 elif self.layerwise:
#                     t_w.data = torch.norm(p.data)*torch.ones_like(p.data).add_(self.eta).data
#                 elif self.filterwise:
#                     if 'conv' in n:
#                         norms = torch.linalg.norm(p.data, dim=[1, 2, 3])  # compute norms
#                         norms = norms.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
#                             [1] + list(p[0, ...].shape))  # bring into shape of param
#                         t_w.data = norms
#                     else:
#                         t_w.abs_().add_(self.eta)
#                 if self.p == '2':
#                     perturbation_layer.mul_(t_w)  # update gradient
#             if self.p == 'infinity':
#                 if ('weight' in n) or self.normalize_bias:
#                     perturbation_layer.sign_().mul_(t_w)
#                 else:
#                     perturbation_layer.sign_()
#             wgrads.append(torch.norm(perturbation_layer, p=2))
#         wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
#
#         n_i=0
#         for n, p in self.model.named_parameters():
#             if (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
#                 continue
#             size_i = list(p.data.shape)
#             perturbation_layer = perturbation[n_i: n_i + numpy.prod(size_i)].view(size_i)
#             n_i += numpy.prod(size_i)
#             t_w = self.state[p].get("eps") # get normalization operator
#             if self.p=='2':
#                 if ('weight' in n) or self.normalize_bias: # second multiplication with t_w
#                     perturbation_layer.mul_(t_w)
#                 eps = t_w
#                 eps[...] = perturbation_layerd[...]
#                 eps.mul_(self.rho / wgrad_norm)
#             elif self.p=='infinity':
#                 eps = t_w
#                 eps[...] = perturbation_layer[...]*self.rho
#             else:
#                 raise NotImplementedError
#             p.add_(eps)
#         self.optimizer.zero_grad()
#
#     @torch.no_grad()
#     def descent_step(self):
#         for n, p in self.model.named_parameters():
#             p.data = self.state[p]['old_p']
#             if (p.grad is None) or (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
#                 continue
#
#         self.optimizer.step()
#         self.optimizer.zero_grad()
#
# class AVG_SAM_BN(AVG_ASAM_BN):
#     @torch.no_grad()
#     def ascent_step(self):
#         perturbation = self.uniform_sphere_numpy(1, self.n_params, cuda=True)[0,...]
#         n_i = 0
#         for n, p in self.model.named_parameters():
#             self.state[p]['old_p'] = p.data.clone()
#             if (self.no_bn and ('norm' in n or 'bn' in n)) or (self.only_bn and 'norm' not in n and 'bn' not in n):
#                 continue
#             size_i = list(p.data.shape)
#             perturbation_layer = perturbation[n_i: n_i + numpy.prod(size_i)].view(size_i)
#             p.add_(self.rho*perturbation_layer.clone().detach())
#             n_i += numpy.prod(size_i)
#         self.optimizer.zero_grad()



def update_grad(optim):
    """
    Applies weight decay + momentum to gradients without updating their parameters
    """
    for group in optim.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for p in group['params']:
            if p.grad is not None:
                state = optim.state[p]
                if 'momentum_buffer' not in state:
                    buf = None
                else:
                    buf = state['momentum_buffer']
                if weight_decay != 0:
                    p.grad.add_(p, alpha=weight_decay)
                if momentum != 0:
                    if buf is None:
                        buf_ = torch.clone(p.grad).detach()
                    else:
                        buf_ = torch.clone(buf).detach()
                        buf_.mul_(momentum).add_(p.grad, alpha=1 - dampening)
                    if nesterov:
                        p.grad.add_(buf_, alpha=momentum)
                    else:
                        p.grad.zero_().add_(buf_)






# shampoo
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]

def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    # use CPU for svd for speed up
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).to(device)


class Shampoo(Optimizer):
    r"""Implements Shampoo Optimizer Algorithm.
    It has been proposed in `Shampoo: Preconditioned Stochastic Tensor
    Optimization`__.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        epsilon: epsilon added to each mat_gbar_j for numerical stability
            (default: 1e-4)
        update_freq: update frequency to compute inverse (default: 1)
    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Shampoo(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ https://arxiv.org/abs/1802.09568
    Note:
        Reference code: https://github.com/moskomule/shampoo.pytorch
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 1,
    ):

        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if epsilon < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if update_freq < 1:
            raise ValueError('Invalid momentum value: {}'.format(momentum))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
        )
        super(Shampoo, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group['momentum']
                weight_decay = group['weight_decay']
                if len(state) == 0:
                    state['step'] = 0
                    if momentum > 0:
                        state['momentum_buffer'] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        # precondition matrices
                        state['precond_{}'.format(dim_id)] = group[
                            'epsilon'
                        ] * torch.eye(dim, out=grad.new(dim, dim))
                        state[
                            'inv_precond_{dim_id}'.format(dim_id=dim_id)
                        ] = grad.new(dim, dim).zero_()

                if momentum > 0:
                    grad.mul_(1 - momentum).add_(
                        state['momentum_buffer'], alpha=momentum
                    )

                if weight_decay > 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # See Algorithm 2 for detail
                for dim_id, dim in enumerate(grad.size()):
                    precond = state['precond_{}'.format(dim_id)]
                    inv_precond = state['inv_precond_{}'.format(dim_id)]

                    # mat_{dim_id}(grad)
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()
                    precond.add_(grad @ grad_t)
                    if state['step'] % group['update_freq'] == 0:
                        inv_precond.copy_(_matrix_power(precond, -1 / order))

                    if dim_id == order - 1:
                        # finally
                        grad = grad_t @ inv_precond
                        # grad: (-1, last_dim)
                        grad = grad.view(original_size)
                    else:
                        # if not final
                        grad = inv_precond @ grad
                        # grad (dim, -1)
                        grad = grad.view(transposed_size)

                state['step'] += 1
                state['momentum_buffer'] = grad
                p.data.add_(grad, alpha=-group['lr'])

        return loss

def precondition_gradient_with_shampoo(optim):
    """Performs preconditioning by directly modifying p.grad without changing buffers
        Parameter itself is not updated!!
    Arguments:
        optim: Shampoo optimizer
    """
    assert isinstance(optim, Shampoo)

    for group in optim.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            order = grad.ndimension()
            original_size = grad.size()
            state = optim.state[p]
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            if len(state) == 0:
                state['step'] = 0
                if momentum > 0:
                    state['momentum_buffer'] = grad.clone()
                for dim_id, dim in enumerate(grad.size()):
                    # precondition matrices
                    state['precond_{}'.format(dim_id)] = group[
                                                             'epsilon'
                                                         ] * torch.eye(dim, out=grad.new(dim, dim))
                    state[
                        'inv_precond_{dim_id}'.format(dim_id=dim_id)
                    ] = grad.new(dim, dim).zero_()

            if momentum > 0:
                grad.mul_(1 - momentum).add_(
                    state['momentum_buffer'], alpha=momentum
                )

            if weight_decay > 0:
                grad.add_(p.data, alpha=group['weight_decay'])

            # See Algorithm 2 for detail
            for dim_id, dim in enumerate(grad.size()):
                precond = state['precond_{}'.format(dim_id)]
                inv_precond = state['inv_precond_{}'.format(dim_id)]

                # mat_{dim_id}(grad)
                grad = grad.transpose_(0, dim_id).contiguous()
                transposed_size = grad.size()
                grad = grad.view(dim, -1)

                grad_t = grad.t()
                precond.add_(grad @ grad_t)
                if state['step'] % group['update_freq'] == 0:
                    inv_precond.copy_(_matrix_power(precond, -1 / order))

                if dim_id == order - 1:
                    # finally
                    grad = grad_t @ inv_precond
                    # grad: (-1, last_dim)
                    grad = grad.view(original_size)
                else:
                    # if not final
                    grad = inv_precond @ grad
                    # grad (dim, -1)
                    grad = grad.view(transposed_size)
            p.grad.zero_().add_(grad)


class TradeSAM(torch.optim.Optimizer):
    def __init__(self, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(TradeSAM, self).__init__(base_optimizer.param_groups, defaults)

        self.optimizer = base_optimizer
        self.param_groups = self.optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def perturb(self):
        # grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] #/ (grad_norm + 1e-7) / (1 - self.weight_dropout)
            for p in group["params"]:
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * torch.randn_like(p.data)*scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group["params"]:
                if not self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
                self.state[p]["e_w"] = 0

    def step(self, zero_grad=True):
        self.optimizer.step()
        if self.zero_grad:
            self.zero_grad()

    # def _grad_norm(self):
    #     shared_device = self.param_groups[0]["params"][
    #         0].device  # put everything on the same device, in case of model parallelism
    #     norm = torch.norm(
    #         torch.stack([
    #             # original sam
    #             # p.grad.norm(p=2).to(shared_device)
    #             # asam
    #             ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
    #             for group in self.param_groups for p in group["params"]
    #             if ((p.grad is not None))
    #         ]),
    #         p=2
    #     )
    #     return norm