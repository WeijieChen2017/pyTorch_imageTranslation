from model import Net


net = Net()
print(net)

# params = list(net.parameters())
# print(len(params))
# print(params[0].size())

# (N,C,D,H,W)
input = torch.randn(4, 1, 32, 32, 32)
out = net(input)
print(out)