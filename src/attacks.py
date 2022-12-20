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


def pgd_attack(model, x, max_iter, **args):
    # pdb.set_trace()
    assert args['rand_init'] == True
    assert (args['norm'] == np.inf or args['norm'] == 2)

    nx = x.clone()

    out = model(nx)
    initial_label = out.max(1)[1]
    pred_nx= out.max(1)[1]
    i_iter = 0
    cumul_dis_2 = 0.
    cumul_dis_inf = 0.
    while pred_nx == initial_label and i_iter < max_iter:
        nx = _pgd(model, x, **args)
        out = model(nx)
        pred_nx = out.max(1)[1]

        eta = (x - nx).cpu()

        i_iter += 1
        cumul_dis_inf = torch.linalg.norm(torch.ravel(eta), ord=np.inf)
        cumul_dis_2 = torch.linalg.norm(eta)
        cumul_dis = {'2': cumul_dis_2, 'inf': cumul_dis_inf}
    return nx, i_iter, cumul_dis


def bim_attack(model, x, **args):
    assert args['rand_init'] == False
    return _pgd(model, x, **args)


def test_bim_attack(model, x, **args):
    assert args['rand_init'] == False
    return _pgd(model, x, **args)


def deepfool_attack(model, x, max_iter, **args):
    """DeepFool attack"""
    nx = x.clone()
    nx.requires_grad_()
    eta = torch.zeros(nx.shape).cuda()

    out = model(nx+eta)
    n_class = out.shape[1]
    initial_label = out.max(1)[1]
    pred_nx = out.max(1)[1]

    i_iter = 0
    cumul_dis_2 = 0.
    cumul_dis_inf = 0.
    while pred_nx == initial_label and i_iter < max_iter:
        out[0, pred_nx].backward(retain_graph=True)
        grad_np = nx.grad.data.clone()
        value_l = np.inf
        w_l = None
        ri = None
        for i in range(n_class):
            if i == initial_label:
                continue

            nx.grad.data.zero_()
            out[0, i].backward(retain_graph=True)
            grad_i = nx.grad.data.clone()

            wi = grad_i - grad_np
            fi = out[0, i] - out[0, initial_label]
            value_i = np.abs(fi.item()) / torch.norm(wi.flatten())

            if value_i < value_l:
                value_l = value_i
                w_l = wi

        ri = value_l/torch.norm(w_l.flatten()) * w_l
        #
        cumul_dis_inf += torch.linalg.norm(torch.ravel(ri.cpu()), ord=np.inf)
        cumul_dis_2 += torch.linalg.norm(ri.cpu())
        #
        eta += ri.clone()
        nx.grad.data.zero_()
        out = model(nx+eta)
        pred_nx = out.max(1)[1]
        i_iter += 1

    cumul_dis = {'2': cumul_dis_2, 'inf': cumul_dis_inf}
    return x+ri, i_iter, cumul_dis


def test_deepfool_attack(model, x, **args):
    """DeepFool attack"""
    nx = x.clone()
    nx.requires_grad_()
    eta = torch.zeros(nx.shape).cuda()

    out = model(nx+eta)
    n_class = out.shape[1]
    initial_label = out.max(1)[1]
    pred_nx = out.max(1)[1]

    i_iter = 0

    while pred_nx == initial_label and i_iter < args['nb_iter']:
        out[0, pred_nx].backward(retain_graph=True)
        grad_np = nx.grad.data.clone()
        value_l = np.inf
        w_l = None
        ri = None
        for i in range(n_class):
            if i == initial_label:
                continue

            nx.grad.data.zero_()
            out[0, i].backward(retain_graph=True)
            grad_i = nx.grad.data.clone()

            wi = grad_i - grad_np
            fi = out[0, i] - out[0, initial_label]
            value_i = np.abs(fi.item()) / torch.norm(wi.flatten())

            if value_i < value_l:
                value_l = value_i
                w_l = wi

        ri = value_l/torch.norm(w_l.flatten()) * w_l
     
        eta += ri.clone()
        nx.grad.data.zero_()
        out = model(nx+eta)
        pred_nx = out.max(1)[1]
        i_iter += 1

    return x+ri
