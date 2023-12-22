import sys
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch.backends.cudnn as cudnn
from data_loader import GetLoader
from torchvision import datasets, transforms
from model import CNNModel
from test import test
import torch
import torch.nn as nn
from torchvision.transforms import v2

source_dataset_name = 'source'
target_dataset_name = 'target'
model_root = 'models'
cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 16
image_size = 256
n_epoch = 30
num_classes = 3
weight_decay = 1e-5

torch.manual_seed(42)

# load data
transform = v2.Compose([
    v2.Resize(image_size),
    v2.RandomCrop(224),
    v2.ToTensor(),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

train_source_set = GetLoader(
    csv_file='/content/drive/MyDrive/Lab/data_classification/labels/train_source_2.csv',
    root_dir='/content/drive/MyDrive/Lab/data_classification/images/train_source_2',
    transform = transform
)

train_target_set = GetLoader(
    csv_file='/content/drive/MyDrive/Lab/data_classification/labels/train_target_2.csv',
    root_dir='/content/drive/MyDrive/Lab/data_classification/images/train_target_2',
    transform = transform    
)

dataloader_source = DataLoader(
    dataset=train_source_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

dataloader_target = DataLoader(
    dataset=train_target_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)


# Create CNNModel instance
my_net = CNNModel()

# setup optimizer
optimizer = optim.Adam(my_net.parameters(), lr=lr, weight_decay=weight_decay)

criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_net = my_net.to(device)
best_accu_t = 0.0

# Training loop
for epoch in range(n_epoch):

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    
    for i in range(len_dataloader):
        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Training model using source data
        s_img, s_label = next(iter(dataloader_source))
        my_net.zero_grad()
        batch_size = len(s_label)
        domain_label = torch.zeros(batch_size).long()

        # Move source data to GPU
        s_img = s_img.to(device, non_blocking=True)
        s_label = s_label.to(device, non_blocking=True)
        domain_label = domain_label.to(device, non_blocking=True)

        class_output, domain_output = my_net(s_img, alpha)
        loss_s_label = criterion(class_output, s_label)
        loss_s_domain = criterion(domain_output, domain_label)

        # Training model using target data
        t_img, _ = next(iter(dataloader_target))
        batch_size = len(t_img)
        domain_label = torch.ones(batch_size).long()

        # Move target data to GPU
        t_img = t_img.to(device)
        domain_label = domain_label.to(device, non_blocking=True)

        _, domain_output = my_net(t_img, alpha)
        loss_t_domain = criterion(domain_output, domain_label)
        loss = loss_t_domain + loss_s_domain + loss_s_label

        # Backward and optimization steps
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], loss_s_label: %f, loss_s_domain: %f, loss_t_domain: %f' \
              % (epoch, i + 1, len_dataloader, loss_s_label.item(),
                 loss_s_domain.item(), loss_t_domain.item()))
        sys.stdout.flush()
      

    torch.save(my_net, '{0}/last.pth'.format(model_root))
    
    print('\n')
    accu_s = test(source_dataset_name)
    print('Accuracy of the %s dataset: %f' % ('synthetic', accu_s))
    accu_t = test(target_dataset_name)
    print('Accuracy of the %s dataset: %f\n' % ('real', accu_t))
    if accu_t > best_accu_t:
        best_accu_s = accu_s
        best_accu_t = accu_t
        torch.save(my_net, '{0}/best.pth'.format(model_root))

print('============ Summary ============= \n')
print('Accuracy of the %s dataset: %f' % ('synthetic', best_accu_s))
print('Accuracy of the %s dataset: %f' % ('real', best_accu_t))
print('Corresponding model was save in ' + model_root + '/best.pth')
