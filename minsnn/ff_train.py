import matplotlib.pyplot as plt 
from model import FF_Net, overlay_y_on_x
import torch 
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Grayscale, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader


if __name__ == "__main__":
    device = torch.device('cuda')
    torch.manual_seed(123)
    data_path = 'data/mnist' 

    transform = Compose([
    Grayscale(),
    ToTensor(),
    Normalize((0,), (1,)),
    Lambda(lambda x: torch.flatten(x))])

    # Train set
    mnist_train = MNIST(data_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)

    # Test set
    mnist_test = MNIST(data_path, train=False, download=True, transform=transform)
    test_loader = DataLoader(mnist_train, batch_size=128, shuffle=False)
    net = FF_Net([784, 500, 500], "lif").to(device) # construct neural net with 784 input neurons, 500 neurons in each of the two hidden layers
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    x_pos = overlay_y_on_x(x, y)  # generate positive samples (i.e., correct labels overlayed on input data)

    # generate negative samples (i.e., incorrect labels overlayed on input data)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])  # random

    # visualize samples
    # for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
    #     visualize_sample(data, name)
    
    loss = net.train(x_pos, x_neg) 
    print('Train error:', 100*(1.0 - net.predict(x).eq(y).float().mean().item()),'%')