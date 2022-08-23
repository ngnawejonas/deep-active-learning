import numpy as np
import torch
import pdb
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method        
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent


def fgsm_attack(model, x, **args):
    return fast_gradient_method(model, x, **args)

def pgd_attack(model, x, **args):
    pdb.set_trace()
    assert args['rand_init'] == True
    assert args['norm'] == np.inf or args['norm'] == 2 
    return projected_gradient_descent(model, x, **args)

def bim_attack(model, x, **args):
    assert args['rand_init'] == False
    return projected_gradient_descent(model, x, **args)

def deepfool_attack(model, x):
    """DeepFool attack"""
    nx =  x.clone()
    nx.requires_grad_()
    out = model(nx)

    n_class = out.shape[1]
    py = out.max(1)[1].item()
   
    out[0, py].backward(retain_graph=True)
    grad_np = nx.grad.data.clone()
    value_l = np.inf
    ri = None
    for i in range(n_class):
        if i == py:
            continue
        nx.grad.data.zero_()
        out[0, i].backward(retain_graph=True)
        grad_i = nx.grad.data.clone()

        wi = grad_i - grad_np
        fi = out[0, i] - out[0, py]
        value_i = np.abs(fi.item()) / torch.norm(wi.flatten())

        if value_i < value_l:
            ri = value_i/torch.norm(wi.flatten()) * wi

    return x + ri