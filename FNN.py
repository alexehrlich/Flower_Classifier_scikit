import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FNN(nn.Module):
    """
        Fully connected feed forward network
    """
    def __init__(self, *args, activation_function=nn.ReLU()):
        super(FNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(args) - 1):
            self.layers.append(nn.Linear(args[i], args[i + 1]))
    
    def forward(self, in_tensor):
        x = in_tensor
        for layer in self.layers:
            x = layer(x)
        return x


class IrisDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train(model, train_loader, device, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    #sets the model to specific mode to prevent overfitting and normalization
    model.train()

    for epoch in range(epochs):
        #Iterate through trainging data:
        for batch_idx, (data, label) in enumerate(train_loader):
            #move tensors to GPU
            data, label = data.to(device), label.to(device)

            #make the forward pass with the current weights and biases
            prediction = model.forward(data)

            #evaluate how good the prediction was
            loss = criterion(prediction, label)

            #reset the accumuation of the gradients of the last batch
            # calculate all derivatiives of the loss with respect to every param
            # make a step with gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Loss: {loss/train_loader.batch_size} in epoch {epoch}")

def evaluate(model, device, test_loader):
    #deactivate the training mode with normalization and stuff
    model.eval()

    #No gradients must be calculated during the 
    with torch.no_grad():
        correct = 0
        total = 0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            outputs = model.forward(data)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load the training data
    data = load_iris()

    X = data.data
    y = data.target

    #normalize the training data
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    #split the training data into train, validation and test
    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

    #Transforming ndarray to tensors for pytorch
    train_dataset = IrisDataset(X_train, y_train)
    test_dataset = IrisDataset(X_test, y_test)

    #DataLoader for batching the trianing data
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    #train the model with the training data and with the validation data
    # Determine the loss and the accuracy with the test set for each epoch
    model = FNN(4, 12, 12, 3)
    train(model, train_loader, device, epochs=100, learning_rate=0.01)

    evaluate(model, device, test_loader)

if __name__ == '__main__':
    main()