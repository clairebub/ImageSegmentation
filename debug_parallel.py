import os

import matplotlib.pyplot as plt
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
        self.fc = nn.Linear(input_size, output_size, bias=True)

    def forward(self, input):
        output = self.fc(input)
        #print(f"In Model: input.shape={input.shape}, output.shape={output.shape}")
        return output


if __name__ == '__main__':

    input_size = 2
    output_size = 4
    num_inputs = 100
    batch_size = 16

    targets = torch.rand(batch_size, output_size)
    dataset = RandomDataset(input_size, num_inputs)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        num_workers=4,
        shuffle=True, 
        drop_last=True)
    model = Model(input_size, output_size)
    model = nn.DataParallel(model)

    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    criterion = nn.MSELoss()
    # # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # print()
    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optim.state_dict():
    #     print(var_name, "\t", optim.state_dict()[var_name])
    # print()

    max_epochs = 3
    for epoch in range(max_epochs):
        print(f"epoch={epoch}")
        if torch.cuda.is_available():
            print(f"on {torch.cuda.current_device()}")
        for inputs in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            model.zero_grad()
            loss.backward()
            optim.step()
    # model = torchvision.models.resnet18(pretrained=True)
    # num_inputs = 100
    # labels = torch.rand(num_inputs, 1000)
    # x = torch.rand(num_inputs, 3, 64, 64)
    # y_hat = model2(x)
    # loss = (y_hat - labels)
    # loss = loss.sum()
    # loss.backward()


 

