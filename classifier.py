import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

def eval_ce(cnn_dict, Z, X, y, cd, key, class_list, n_samples, n_split):
    # given dictionary containing trained CNN for each class, calculate the cross entropy loss
    # the key of cnn_dict represents which class it is trained for 

    n_class = len(class_list)
    n_eval = int((n_samples - n_split)/(n_class-1))
    CELoss=0
    for l in class_list:
        X_real = X[np.nonzero(y==l)[0]]
        y_real = np.zeros((X_real.shape[0]))

        X_fake = []
        for m in class_list:
            if m != l:
                idxm = np.nonzero(y==m)[0]
                if key in ['GB', 'NB', 'GBNB','DD','tree']:
                    Xflip_m = cd.inverse_transform(Z[idxm],np.ones(Z[idxm].shape[0])*l )
                else:
                    Xflip_m = cd.inverse(Z[idxm],np.ones(Z[idxm].shape[0])*l )
                X_fake.append(Xflip_m[:n_eval])

        X_fake = np.array(X_fake).reshape(-1, X.shape[1])
        y_fake = np.ones(X_fake.shape[0])
        CELoss += CNN_CELoss(cnn_dict[l],np.concatenate((X_real, X_fake)), np.concatenate((y_real, y_fake)))
    CELoss = CELoss/n_class
    return CELoss


def train_CNN(X, y, Net, num_epochs=30, batch_size=64, lr=1e-4,
                beta1=0.5, random_seed=0, device =torch.device('cpu'), verbosity=0):
    start = time.time()
    def init_weights(m):
        classname = m.__class__.__name__
        if classname.find('conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def preprocess(Xi, yi):
        Xi = Xi.reshape(Xi.shape[0], 1, 28, 28) # Convert to images
        Xi = torch.Tensor(Xi)
        yi = torch.Tensor(yi)
        return Xi, yi

    X,y  = preprocess(X, y)

    net = Net().to(device)
    #net.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))

    def create_dataloader(Xi, yi):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xi, yi),
            batch_size=batch_size, shuffle=True)
    dataloader = create_dataloader(X, y)

    # Lists to keep track of progress
    losses = []
    accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        n_correct = 0
        n_total = 0
        loss = 0
        for i, (Xb, yb) in enumerate(dataloader, 0):
            net.zero_grad()
            Xb = Xb.to(device)
            yb = yb.to(device)
            output = net(Xb)
            loss = criterion(output, yb.long())
            loss.backward()
            optimizer.step()
            
            # To keep track of accuracies
            n_batch = output.shape[0]
            total_loss = loss * n_batch 
            _, y_pred = torch.max(output,1)
            n_correct += (y_pred == yb).float().sum()
            n_total += output.shape[0]

    def get_stats(Xi, yi):
        Xi = Xi.to(device)
        yi = yi.to(device)
        outputi = net(Xi)
        loss = criterion(outputi, yi.long())
        _, y_predi = torch.max(outputi,1)
        n_correcti = (y_predi == yi).float().sum()
        acc = n_correcti/Xi.shape[0]
        def simplify(a):
            return float(a.detach().cpu().numpy())
        return simplify(loss), simplify(acc)
    train_loss, train_acc = get_stats(X, y)
    if verbosity >0:
        print(f'time {time.time()-start} s')
        print('train_loss:', train_loss)
        print('train_acc:', train_acc)
    return net


def CNN_CELoss(model, X, y, device=torch.device('cpu'), verbosity=0, acc=False):
    # find the cross entropy loss given a trained CNN

    X = torch.Tensor(X)
    y = torch.Tensor(y)

    def preprocess(Xi, yi):
        Xi = Xi.reshape(Xi.shape[0], 1, 28, 28) # Convert to images
        Xi = torch.Tensor(Xi)
        yi = torch.Tensor(yi)
        return Xi, yi


    X,y  = preprocess(X, y)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    def get_stats(Xi, yi):
        Xi = Xi.to(device)
        yi = yi.to(device)
        outputi = model(Xi)
        loss = criterion(outputi, yi.long())
        _, y_predi = torch.max(outputi,1)
        n_correcti = (y_predi == yi).float().sum()
        acc = n_correcti/Xi.shape[0]
        def simplify(a):
            return float(a.detach().cpu().numpy())
        return simplify(loss), simplify(acc)
    test_loss, test_acc = get_stats(X, y)
    if verbosity >0:
        print('test_loss:', test_loss)
        print('test_acc:', test_acc)

    if acc:
        return test_loss, test_acc
    else:
        return test_loss

    
class CNN_MNIST(nn.Module):

    # CNN classifier for MNIST from https://www.kaggle.com/dsaichand3/mnist-classifier-in-pytorch

    def __init__(self):
        super(CNN_MNIST,self).__init__()
        self.cnn_1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride=1, padding=0)
        self.cnn_2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        
        self.fc1 = nn.Linear(32 * 4 * 4, 128) 
        self.fc2 = nn.Linear(128, 64) 
        self.out = nn.Linear(64, 10) 
        
    def forward(self,x):
        
        out = self.cnn_1(x)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)
        
        out = self.cnn_2(out)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.out(out)
        
        return out


class CNN_FMNIST(nn.Module):

    # CNN classifier for FMNIST from https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy
    
    def __init__(self):
        super(CNN_FMNIST, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out