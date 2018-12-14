
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

show = ToPILImage() # change tensor to image

# data transfrom
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

# train datasets
trainset = tv.datasets.CIFAR10(
    root='/media/luo/result/',
    train=True,
    download=False,
    transform=transforms)

trainloader = DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

# test datasets
testset = tv.datasets.CIFAR10(
    root='/media/luo/result/',
    train=False,
    download=False,
    transform=transforms)

testloader = DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

######### show samples #########
# (1)dataloader:
# dataiter = iter(trainloader)
# images, label = dataiter.next()
# print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
# show(tv.utils.make_grid((images + 1) / 2)).resize(100, 100)
#
# (2)dataset:
# (data, label) = trainset[100]
# print(calsses[label])
# show((data + 1) / 2).resize(100, 100)