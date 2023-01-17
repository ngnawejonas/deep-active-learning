import numpy as np
import tqdm
import torch
from cifar10_autoattack import ResNet18, get_CIFAR10, set_seeds
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent as pgd

#Load
model = ResNet18()
model.load_state_dict(torch.load('./cifarmodel'), strict=False)

batch_size = 128
#
train_dataset, test_dataset = get_CIFAR10(45000)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels, idx in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# epsilon = 0.3
# adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', verbose=False)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
  
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in tqdm(test_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#         x_adv = adversary.run_standard_evaluation(images, labels)
#         outputs = model(x_adv)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))



# !pip install cleverhans


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
pgd_params = {'eps': 8/255., 'eps_iter': 0.005, 'nb_iter': 50, 'norm': np.inf, 'targeted': False, 'rand_init': True}

correct = 0
total = 0
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = 128,
                                           shuffle = True)
for images, labels, idx in tqdm(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    x_adv = pgd(model, images, y=labels, **pgd_params)
    outputs = model(x_adv)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('AAccuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# # import matplotlib.pyplot as plt
# def show(im, predicted, label, txt):
# # im1 = images[0][0].cpu().numpy()
#   plt.figure()
#   im = im[0].detach().cpu().numpy()
#   plt.imshow(im)
#   # print(predicted.cpu().numpy(), label.cpu().numpy(), (label==predicted).cpu().numpy())
#   title = txt+' {} {} {}'.format(predicted.cpu().numpy(), label.cpu().numpy(), (label==predicted).cpu().numpy())
#   plt.title(title)

# from re import X
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
set_seeds(42)
pgd_params = {'eps': 0.3, 'eps_iter': 0.1, 'nb_iter': 20, 'norm': np.inf, 'targeted': False, 'rand_init': True}

correct = 0
xcorrect = 0
total = 0
distances = []
xtest_dataset = torch.utils.data.Subset(test_dataset, np.arange(10))
test_loader = torch.utils.data.DataLoader(dataset = xtest_dataset,
                                           batch_size = 1,
                                           shuffle = False)  
for images, labels in tqdm(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    x_adv = pgd(model, images, **pgd_params)
    outputs = model(images)
    xoutputs = model(x_adv)
    _, predicted = torch.max(outputs.data, 1)
    _, xpredicted = torch.max(xoutputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    xcorrect += (xpredicted == labels).sum().item()
    # print()
    # show(images, predicted, labels, 'clean')
    # show(x_adv, xpredicted, labels, 'adv')
    # print('---')
print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
print('AAccuracy of the network on the 10000 test images: {} %'.format(100 * xcorrect / total))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
pgd_params = {'eps': 0.3, 'eps_iter': 0.05, 'nb_iter': 50, 'norm': np.inf, 'targeted': False, 'rand_init': True}

correct = 0
total = 0
for images, labels in tqdm(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    x_adv = pgd(model, images, y=labels, **pgd_params)
    outputs = model(x_adv)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
pgd_params = {'eps': 0.3, 'eps_iter': 0.05, 'nb_iter': 100, 'norm': np.inf, 'targeted': False, 'rand_init': True}

correct = 0
total = 0
for images, labels in tqdm(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    x_adv = pgd(model, images, y=labels, **pgd_params)
    outputs = model(x_adv)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Test the model
# # In test phase, we don't need to compute gradients (for memory efficiency)
# pgd_params = {'eps': 0.3, 'eps_iter': 0.1, 'nb_iter': 50, 'norm': np.inf, 'targeted': False, 'rand_init': True}
# 
# correct = 0
# total = 0
# 
# test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
#                                            batch_size = 1,
#                                            shuffle = True)
# for images, labels in tqdm(test_loader):
#     if total > 300:
#       break
#     images, labels = images.to(device), labels.to(device)
#     x_adv = pgd(model, images, y=labels, **pgd_params)
#     outputs = model(x_adv)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum().item()
#     
# print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))