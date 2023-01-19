import numpy as np
import torch
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent as _pgd
from pgd_adaptive import projected_gradient_descent as adaptive_pgd
# from autoattack import AutoAttack

def compute_norm(x, norm):
    with torch.no_grad():
        if norm == np.inf:
            return torch.linalg.norm(torch.ravel(x.cpu()), ord=np.inf).numpy()
        elif norm == 2:
            return torch.linalg.norm(x.cpu()).numpy()
        else:
            raise NotImplementedError


# def test_auto_attack(model, x, **args):
#     adversary = AutoAttack(model, **args)
#     x_adv = adversary.run_standard_evaluation(x, labels, bs=batch_size)
#     return x_adv


def test_pgd_attack(model, x, y=None, **args):
    # pdb.set_trace()
    assert args['rand_init'] == True
    assert (args['norm'] == np.inf or args['norm'] == 2)
    return _pgd(model, x, y=y, **args)


def pgd_attack(model, x, max_iter, **args):
    args['nb_iter'] = max_iter
    return adaptive_pgd(model, x, **args)
    
# def pgd_attack(model, x, max_iter, **args):
#     # pdb.set_trace()
#     assert args['rand_init'] == True
#     assert (args['norm'] == np.inf or args['norm'] == 2)

#     nx = x.clone()

#     out = model(nx)
#     initial_label = out.max(1)[1]
#     pred_nx= out.max(1)[1]
#     i_iter = 0
#     cumul_dis_2 = 0.
#     cumul_dis_inf = 0.
#     while pred_nx == initial_label and i_iter < max_iter:
#         nx = _pgd(model, x, **args)
#         out = model(nx)
#         pred_nx = out.max(1)[1]

#         eta = (x - nx).cpu()

#         i_iter += 1
#         cumul_dis_inf += torch.linalg.norm(torch.ravel(eta), ord=np.inf)
#         cumul_dis_2 += torch.linalg.norm(eta)
#     cumul_dis = {'2': cumul_dis_2, 'inf': cumul_dis_inf}
#     return nx, i_iter, cumul_dis

def test_bim_attack(model, x, y, **args):
    # pdb.set_trace()
    assert args['rand_init'] == False
    assert (args['norm'] == np.inf or args['norm'] == 2)
    return _pgd(model, x, y=y, **args)


def bim_attack(model, x, max_iter, **args):
    args['nb_iter'] = max_iter
    assert args['rand_init'] == False
    return adaptive_pgd(model, x, **args)


def deepfool_attack(model, x, max_iter, **args):
    """DeepFool attack"""
    nx = x.clone()
    nx.requires_grad_()
    eta = torch.zeros(nx.shape).cuda() if torch.cuda.is_available() else torch.zeros(nx.shape)

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
        for i in range(n_class):
            if i == initial_label:
                continue

            nx.grad.data.zero_()
            out[0, i].backward(retain_graph=True)
            grad_i = nx.grad.data.clone()

            wi = grad_i - grad_np
            fi = out[0, i] - out[0, initial_label]
            value_i = np.abs(fi.item()) / torch.norm(wi.flatten())
            # breakpoint()
            if value_i < value_l:
                value_l = value_i
                w_l = wi

        ri = value_l/torch.norm(w_l.flatten()) * w_l
        #
        cumul_dis_inf += compute_norm(ri, norm=np.inf)
        cumul_dis_2 += compute_norm(ri, norm=2)
        #
        eta += ri.clone()
        nx.grad.data.zero_()
        out = model(nx+eta)
        pred_nx = out.max(1)[1]
        i_iter += 1

    cumul_dis = {'2': cumul_dis_2, 'inf': cumul_dis_inf}
    return x+ri, i_iter, cumul_dis


def test_deepfool_attack(model, x, y=None, **args):
    """DeepFool attack"""
    return deepfool_attack(model, x, max_iter=args['nb_iter'])[0]