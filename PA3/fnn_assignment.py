import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

# Function for loading notMNIST Dataset
def loadData(datafile = "notMNIST.npz"):
    with np.load(datafile) as data:
        Data, Target = data["images"].astype(np.float32), data["labels"]
        np.random.seed(7)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Custom Dataset class.
class notMNIST(Dataset):
    def __init__(self, annotations, images, transform=None, target_transform=None):
        self.img_labels = annotations
        self.imgs = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

#Define FNN
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()

        #TODO
        #DEFINE YOUR LAYERS HERE
        self.layer1 = nn.Linear(784, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        
    def forward(self, x):
        
        #TODO
        #DEFINE YOUR FORWARD FUNCTION HERE
        x_flattened = torch.flatten(x, start_dim=1)
        
        x_layer1 = F.relu(self.layer1(x_flattened))
        x_layer2 = F.relu(self.layer2(x_layer1))
        x_output = self.layer3(x_layer2)

        return x_output
    
    
# Commented out IPython magic to ensure Python compatibility.
# Compute accuracy
def get_accuracy(model, dataloader):

    model.eval()
    device = next(model.parameters()).device
    accuracy = 0.0
    total = 0.0
    correct = 0.0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            # TODO
            output = model(images)
            pred = output.max(1, keepdim=True)[1]
            
            accuracy += pred.eq(labels.view_as(pred)).sum().item()
            total += images.shape[0]

    # Return the accuracy
    return accuracy / total

def train(model, device, learning_rate, train_loader, val_loader, test_loader, num_epochs=50, verbose=False):
    
    #TODO
    # Define your cross entropy loss function here
    # Use cross entropy loss
    criterion = nn.CrossEntropyLoss()

    #TODO
    # Define your optimizer here
    # Use AdamW optimizer, set the weights, learning rate argument.
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    acc_hist = {'train':[], 'val':[], 'test': []}

    for epoch in range(num_epochs):
        model = model.train()
        
        ## training step
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            # TODO
            # Follow the step in the tutorial
            ## forward + backprop + loss
            out = model(images)
            loss = criterion(out, labels)

            ## update model params
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        acc_hist['train'].append(get_accuracy(model, train_loader))
        acc_hist['val'].append(get_accuracy(model, val_loader))
        acc_hist['test'].append(get_accuracy(model, test_loader))

    if verbose:
      print('Epoch: %d | Train Accuracy: %.2f | Validation Accuracy: %.2f | Test Accuracy: %.2f' \
          %(epoch, acc_hist['train'][-1], acc_hist['val'][-1], acc_hist['test'][-1]))

    return model, acc_hist

def experiment(learning_rate=0.0001, num_epochs=50, verbose=False):
    # Use GPU if it is available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Inpute Batch size:
    BATCH_SIZE = 32

    # Convert images to tensor
    transform = transforms.Compose(
        [transforms.ToTensor()])

    # Get train, validation and test data loader.
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    train_data = notMNIST(trainTarget, trainData, transform=transform)
    val_data = notMNIST(validTarget, validData, transform=transform)
    test_data = notMNIST(testTarget, testData, transform=transform)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Specify which model to use
    model = FNN()

    # Loading model into device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    model, acc_hist = train(model, device, learning_rate, train_loader, val_loader, test_loader, num_epochs=num_epochs, verbose=verbose)

    # Release the model from the GPU (else the memory wont hold up)
    model.cpu()

    return model, acc_hist



def compare_lr():
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    hist = []

    for lr in learning_rates:
        _, acc_hist = experiment(learning_rate=lr, num_epochs=50, verbose=False)
        hist.append(acc_hist)
    
    plt.figure(figsize=(10, 6))
    
    for i, lr in enumerate(learning_rates):
        plt.plot(hist[i]['train'], label=f'LR={lr}', linestyle='--')
        plt.plot(hist[i]['test'], label=f'LR_test={lr}', linestyle='-')
        
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    compare_lr()
