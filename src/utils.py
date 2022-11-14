from attacks import test_pgd_attack, pgd_attack, bim_attack, fgsm_attack, deepfool_attack
# from pgd_adaptive import pgd_attack

def log_to_file(file_name, line):
    filepath = 'results/'+file_name
    file = open(filepath, 'a')
    file.write(line)
    if not line.endswith('\n'):
        file.write('\n')
    file.close()

def get_attack_fn(name='fgsm'):
    if name == 'fgsm':
        return fgsm_attack
    elif name == 'bim':
        return bim_attack
    elif name == 'pgd':
        return pgd_attack
    elif name == 'test_pgd':
        return test_pgd_attack
    elif name == 'deepfool':
        return deepfool_attack
    else:
        raise NotImplementedError('Attack "{}" not implemented'.format(name))