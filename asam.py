import torch.nn as nn
import torch
from collections import defaultdict
from collections import defaultdict
class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01, adaptive=True, p='2', normalize_bias=False, elementwise=True, filterwise=False, layerwise=False, layerwise_david=False, exclude=[]):
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
        self.layerwise_david=layerwise_david
        self.exclude=exclude

        if self.adaptive:
            assert sum([self.elementwise,self.filterwise, self.layerwise, self.layerwise_david])==1
        else:
            assert sum([self.elementwise,self.filterwise, self.layerwise, self.layerwise_david])==0

    @torch.no_grad()
    def ascent_step(self):
        if self.adaptive:
            if self.layerwise_david:
                for n,p in self.model.named_parameters():
                    if p.grad is None:
                        continue
                    self.state[p]['old_p'] = p.data.clone()
                    exclude = False
                    for e in self.exclude:
                        if n.find(e) >= 0:
                            exclude = True
                    if not exclude:
                        e_w = torch.norm(p) * p.grad * self.rho / (
                                    torch.norm(p.grad) + 1e-12)  # new, overall perturbation is not of size rho
                        p.add_(e_w)
            else:
                wgrads = []
                for n, p in self.model.named_parameters():
                    self.state[p]['old_p'] = p.data.clone()

                    exclude = False
                    for e in self.exclude:
                        if n.find(e) >= 0:
                            exclude = True
                    if p.grad is None or exclude:
                        continue

                    t_w = self.state[p].get("eps")
                    if t_w is None: # initialize t_w, different for all cases
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
                    exclude = False
                    for e in self.exclude:
                        if n.find(e) >= 0:
                            exclude = True
                    if p.grad is None or exclude:
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
        else:
            grads = []
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
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
