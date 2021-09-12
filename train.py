from model import Net
import torch


net = Net()
print(net)

# params = list(net.parameters())
# print(len(params))
# print(params[0].size())

# (N,C,D,H,W)
input = torch.randn(4, 1, 64, 64, 64)
out = net(input)
print(out.size())