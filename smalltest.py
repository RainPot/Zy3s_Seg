import torch
import torch.nn as nn
import torch.nn.functional as F

input1 = torch.randn(2, 64, 128, 128)
input2 = torch.randn(2, 64, 128 ,128)
output = F.pairwise_distance(input1, input2, p=2, keepdim=True)
output = output.view(2, 1, 128 * 128)
print(output.size())
print(output)
Sigmoid = nn.Softmax(dim=2)
output = Sigmoid(output)
output = output.view(2, 1, 128, 128)
print(output)
input3 = torch.randn(2, 64, 128, 128)
c = input3 * output
print(c.size())

for i in range(1):
    print('!!!!')