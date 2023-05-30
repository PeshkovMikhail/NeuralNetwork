import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mnist

class Mnist(torch.utils.data.Dataset):
    def __init__(self, train = True):
        if train:
            data, labels, _, _ = mnist.load()
        else:
            _, _, data, labels = mnist.load()
        print(data.shape)
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float()/255, self.labels[idx]  

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net() # Формируем модель НСpass
optim = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

trainloader = torch.utils.data.DataLoader(Mnist(), batch_size=32, shuffle = True)
testloader = torch.utils.data.DataLoader(Mnist(False), batch_size=1)

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optim.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
results = np.zeros((3, 10))
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients

        # forward + backward + optimize
        outputs = model(inputs)
        results[0][labels[0]] += 1
        pr = torch.argmax(outputs, dim=1)
        if(pr == labels):
            results[1][labels[0]] +=1
        results[2][pr[0]] += 1
print(results[0])
print(results[1])
print(results[2])
print(results[1]/results[0])