import numpy as np
import torch
import pdb      
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent as _pgd
# from autoattack import AutoAttack

from pgd_adaptive import projected_gradient_descent as adapted_pgd

# def test_auto_attack(model, x, **args):
#     adversary = AutoAttack(model, **args)
#     x_adv = adversary.run_standard_evaluation(x, labels, bs=batch_size)
#     return x_adv

def test_pgd_attack(model, x, **args):
    # pdb.set_trace()
    assert args['rand_init'] == True
    assert (args['norm'] == np.inf or args['norm'] == 2)
    return _pgd(model, x, **args)

def pgd_attack(model, x, **args):
    # pdb.set_trace()
    assert args['rand_init'] == True
    assert (args['norm'] == np.inf or args['norm'] == 2)
    return adapted_pgd(model, x, **args)

def bim_attack(model, x, **args):
    assert args['rand_init'] == False
    return _pgd(model, x, **args)

def test_bim_attack(model, x, **args):
    assert args['rand_init'] == False
    return _pgd(model, x, **args)

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
        w_l = None
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
                value_l = value_i
                w_l = wi

        ri = value_l/torch.norm(w_l.flatten()) * w_l
        eta += ri.clone()
        nx.grad.data.zero_()
        out = model(nx+eta)
        py = out.max(1)[1].item()
        i_iter += 1
    
        cumul_dis_inf += torch.linalg.norm(torch.ravel(ri.cpu()), ord=np.inf)
        cumul_dis_2 += torch.linalg.norm(ri.cpu())
        cumul_dis = {'2': cumul_dis_2, 'inf':cumul_dis_inf}
    return x+ri, i_iter, cumul_dis

def test_deepfool_attack(model, x, **args):
    """DeepFool attack"""
    nx = x.clone()
    nx.requires_grad_()
    eta = torch.zeros(nx.shape).cuda()

    out = model(nx+eta)
    n_class = out.shape[1]
    py = out.max(1)[1].item()
    ny = out.max(1)[1].item()

    i_iter = 0

    while py == ny and i_iter < args['max_iter']:
        out[0, py].backward(retain_graph=True)
        grad_np = nx.grad.data.clone()
        value_l = np.inf
        w_l = None
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
                value_l = value_i
                w_l = wi

        ri = value_l/torch.norm(w_l.flatten()) * w_l
        eta += ri.clone()
        nx.grad.data.zero_()
        out = model(nx+eta)
        py = out.max(1)[1].item()
        i_iter += 1

    return x+ri, i_iter, cumul_dis