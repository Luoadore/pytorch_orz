import torch.nn as nn
import torch as t

t.set_num_threads(8)

from lenet import LeNet
net = LeNet()

# loss and optim
from torch import optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training
for epoch in range(2):

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data

        # grad to zero
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # weight
        optimizer.step()

        # log
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' \
                  % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('training done.')