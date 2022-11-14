import numpy as np
import torch
import pdb
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method        
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent as unedited_pgd

from pgd_adaptive import projected_gradient_descent as pgd


def fgsm_attack(model, x, **args):
    return fast_gradient_method(model, x, **args)

def test_pgd_attack(model, x, **args):
    # pdb.set_trace()
    assert args['rand_init'] == True
    assert (args['norm'] == np.inf or args['norm'] == 2)
    return unedited_pgd(model, x, **args)

def pgd_attack(model, x, **args):
    # pdb.set_trace()
    assert args['rand_init'] == True
    assert (args['norm'] == np.inf or args['norm'] == 2)
    return pgd(model, x, **args)

def bim_attack(model, x, **args):
    assert args['rand_init'] == False
    return pgd(model, x, **args)

def deepfool_attack(model, x, **args):
    """DeepFool attack"""
    nx = x.clone()
    nx.requires_grad_()
    eta = torch.zeros(nx.shape).cuda()

    out = model(nx+eta)
    n_class = out.shape[1]
    py = out.max(1)[1].item()
    ny = out.max(1)[1].item()

    i_iter = 0
    cumul_dis_2 = 0.
    cumul_dis_inf = 0.
    while py == ny and i_iter < args['max_iter']:
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
            value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())

            if value_i < value_l:
                ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi
                value_l = value_i

        eta += ri.clone()
        nx.grad.data.zero_()
        out = model(nx+eta)
        py = out.max(1)[1].item()
        i_iter += 1
        cumul_dis_inf += torch.linalg.norm(torch.ravel(ri.cpu()), ord=np.inf)
        cumul_dis_2 += torch.linalg.norm(ri.cpu()) 
    return x+ri, i_iter, cumul_dis_2, cumul_dis_inf