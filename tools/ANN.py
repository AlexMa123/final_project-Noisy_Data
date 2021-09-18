#%%
from torch import nn
from torch import optim
import torch

linear_network = nn.Sequential(
    nn.Linear(400, 25),
    nn.ReLU(),
    nn.Linear(25, 10)
)


def set_weights(model, weights, bias):
    assert weights[0].shape == model[0].weight.shape
    assert weights[1].shape == model[2].weight.shape
    assert bias[0].shape == model[0].bias.shape
    assert bias[1].shape == model[2].bias.shape

    model[0].weight = nn.Parameter(weights[0])
    model[2].weight = nn.Parameter(weights[1])
    model[0].bias = nn.Parameter(bias[0])
    model[2].bias = nn.Parameter(bias[1])


def potential(model, data_X, data_Y, Lambda=0):
    loss_fn = nn.CrossEntropyLoss()
    predict_Y = model(data_X)
    loss = loss_fn(predict_Y, data_Y)
    if Lambda != 0:
        regulation = 0
        for p in model.parameters():
            regulation += torch.sum(torch.abs(p.data))
        loss = loss + Lambda * regulation
    return loss


def train_network(model, train_dataset, test_dataset, MaxIter, lr=0.01, Lambda=0):
    opt = optim.SGD(model.parameters(), lr)
    losses = []
    test_loss = []
    data_X, data_Y = train_dataset
    test_X, test_Y = test_dataset
    for epoch in range(MaxIter):
        opt.zero_grad()
        loss = potential(model, data_X, data_Y, Lambda)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        test_loss.append(potential(model, test_X, test_Y, Lambda=0).item())
    return losses, test_loss

#%%
# def cost_function(model, data_X, data_Y):
    # Y_predict = 

if __name__ == '__main__':
    weights = [torch.ones((25, 400)), torch.ones((10, 25))]
    bias = [torch.ones(25), torch.ones(10)]
    set_weights(linear_network, weights, bias)
    for p in linear_network.parameters():
        print(torch.sum(p.data))
# %%
