import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets
from torch.utils.data import Dataset,DataLoader

def test(dataset_name):
    assert dataset_name in ['source', 'target']

    model_root = 'models'
#    image_root = os.path.join('dataset', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 8
    image_size = 224
    alpha = 0

    """load data"""

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    if dataset_name == 'source':
        dataset = GetLoader(
            csv_file='/content/drive/MyDrive/Lab/data_classification/labels/test_source_2.csv',
            root_dir='/content/drive/MyDrive/Lab/data_classification/images/test_source_2',
            transform = transform
        )
    else: 
        dataset = GetLoader(
            csv_file='/content/drive/MyDrive/Lab/data_classification/labels/test_target_2.csv',
            root_dir='/content/drive/MyDrive/Lab/data_classification/images/test_target_2',
            transform = transform
        )        
      

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ test """

    my_net = torch.load(os.path.join(
        model_root, 'last.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = next(data_target_iter)
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        class_output, _ = my_net(input_data=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
