import os

import torch 
import torch.nn as nn 
import torchvision

class RandomDataset(torch.utils.data.Dataset):

    def __init__(self, input_size, num_inputs):
        self.len = num_inputs
        self.data = torch.randn(num_inputs, input_size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
    

class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        #print(f"In Model: input.shape={input.shape}, output.shape={output.shape}")
        return output


if __name__ == '__main__':

    input_size = 3
    output_size = 4
    num_inputs = 100
    batch_size = 16

    dataset = RandomDataset(input_size, num_inputs)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        num_workers=4,
        shuffle=True, 
        drop_last=False)
    model = Model(input_size, output_size)
    model = nn.DataParallel(model)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optim.step()

    for x in data_loader:
        print(f'x.shape={x.shape}')
        y_hat = model(x)
        print(f'y_hat.shape={y_hat.shape}')

    # model = torchvision.models.resnet18(pretrained=True)
    # num_inputs = 100
    # labels = torch.rand(num_inputs, 1000)
    # x = torch.rand(num_inputs, 3, 64, 64)
    # y_hat = model2(x)
    # loss = (y_hat - labels)
    # loss = loss.sum()
    # loss.backward()


 

