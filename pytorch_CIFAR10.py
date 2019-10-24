import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# define the transforms to use
# creates a tensor with all values divided in half to normalize

train_transforms = transforms.Compose(
    [transforms.RandomRotation(30),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Create training and testing datasets
train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transforms)
# loader creates an iterable object for later use
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4,
                                          shuffle=True, num_workers=0)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4,
                                         shuffle=False, num_workers=0)

num_classes = 10

# Define the model to use

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # the functional nn allows you to get more explicit than nn.sequential
        # which means you have to manually define the parameters
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)

        self.fc1 = nn.Linear(128 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    # Define the forward pass, overwrite X for different layers
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # .view behaves like -1 in numpy.reshape(), i.e. the actual value for this dimension will
        # be inferred so that the number of elements in the view matches the original number of elements.
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# instantiate the class model

model = Net()

# define the loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# to visualize model
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)

# build writer
writer = SummaryWriter()
writer.add_image('images', grid)
writer.add_graph(model, images)

# train the network

for epoch in range(65):

    running_loss = 0.0
    total_correct = 0
    total = 0

    # for every instance/example in the train_loader, get data and index
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # be sure to zero the gradients for every new epoch
        optimizer.zero_grad()

        # Instantiate forward + backward + optimize
        # define the outputs as a function of the net on inputs
        outputs = model(inputs)
        # set the loss as difference between labels and outputs with chosen criterion
        loss = criterion(outputs, labels)
        # Carry out backprop
        loss.backward()
        # Optimize
        optimizer.step()

        # Get predicted
        _, predicted = torch.max(outputs.data, 1)
        # total is the size of labels, every label
        total += labels.size(0)
        # correct is where predicted is equal to label value
        total_correct += (predicted == labels).sum().item()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # make calls to summary writer
    writer.add_scalar('Loss', running_loss, epoch)
    writer.add_scalar('Number Correct', total_correct, epoch)
    writer.add_scalar('Accuracy', 100 * total_correct / total)

print('Finished Training')

# use the test_loader to get images and labels for the test set

data_iter = iter(test_loader)
images, labels = data_iter.next()

writer.close()

def imshow(img):
    # because we normalized the data, we need to put it back to its original state
    img = img / 2 + 0.5
    # create a numpy array out of the image
    img = img.numpy()
    # transpose image to format - (height, width, color)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

# show images
imshow(torchvision.utils.make_grid(images))

# define all the classes we have in the dataset

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# print output
print('Actual: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# need to make the outputs just the images run through the network
outputs = model(images)

# getting the most likely class for the prediction
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# check to see how the network as a whole performs

correct = 0
total = 0

#  "with torch.no_grad()" temporarily sets all the requires_grad flag to false
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # total is the size of labels, every label
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Model accuracy on test data: %d %%' % (100 * correct / total))
