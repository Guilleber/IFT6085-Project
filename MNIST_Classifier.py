import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from torchvision import datasets, transforms


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ELU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ELU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 256, 5),
            nn.ELU()
        )
        
        self.linear = nn.Linear(256, 10)
        return
        
        
    def forward(self, x):
        return self.linear(self.convs(x).view(-1, 256))
        
        
class g(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(28*28,10)
        self.a = nn.Parameter(torch.zeros(10).uniform_(-1, 1))
        self.a.requires_grad = True
        
        
    def forward(self, x):
        return self.a*torch.exp(self.W(x.view(-1, 28*28)))
        
        
def train(train_loader, use_cuda=False, modified_loss=False):
    model = Classifier()
    if use_cuda:
        model = model.cuda()
    model.train()
    if modified_loss:
        g_ = g()
        lambda_ = 10
        if use_cuda:
            g_ = g_.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(20):
        for idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            out = model(data)
            if modified_loss:
                out += g_(data)
            loss = F.cross_entropy(out, target)
            if modified_loss:
                loss += lambda_*torch.norm(g_.a)**2
            loss.backward()
            optimizer.step()
            if idx%100 == 0:
                print(loss.data.cpu().numpy())
    return model
    
    
def test(test_loader, model, use_cuda=False):
    model.eval()
    sum_loss = 0
    sum_correct = 0
    nb = 0
    for data, target in test_loader:
        if use_cuda:
            nb += 1
            data = data.cuda()
            target = target.cuda()
            out = model(data)
            sum_loss += F.cross_entropy(out, target).data.cpu().numpy()
            pred = out.argmax(dim=1, keepdim=True)
            sum_correct += pred.eq(target.view_as(pred)).mean().data.cpu().numpy()
            
    mean_loss = sum_loss/nb
    acc = sum_correct/nb
    
    return (mean_loss, acc)
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="MNIST Classifier")
    parser.add_argument('--use_cuda', type=bool, default=False)
    parser.add_argument('--modified_loss', type=bool, default=False)
    args = parser.parse_args()
    torch.manual_seed(10)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True, **kwargs)
        
    model = train(train_loader, use_cuda=args.use_cuda, modified_loss=args.modified_loss)
    (loss, acc) = test(test_loader, model, use_cuda=args.use_cuda)