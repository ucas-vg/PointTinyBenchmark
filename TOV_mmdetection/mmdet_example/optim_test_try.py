import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(3, 2)
        self.bn = torch.nn.BatchNorm1d(2)

    def forward(self, x):
        return self.bn(self.w(x))


def loss_fn(p, q):
    return ((p - q) ** 2).mean()


def update_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def fix_norm(m):
    from torch.nn.modules.batchnorm import _NormBase
    if isinstance(m, _NormBase):
        m.eval()


def fix_model(model, optimizer):
    update_lr(optimizer, 0)
    model.apply(fix_norm)


# loss设成0不能完全没有梯度, 因为还有momentum buffer和weight_decay
# 只要lr设置为0，即使momentum和weight_decay都不为0， 网络就不会有梯度更新了，但是BN的running mean和running var

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1)
for p in model.parameters():
    print(p.data)
print(model.bn.running_mean)
print()

x = torch.FloatTensor([[4, 5, 6], [1, 2, 3.]])

optimizer.zero_grad()
loss_fn(model(x), 1).backward()
optimizer.step()

for p in model.parameters():
    print(p.data)
print(model.bn.running_mean)
print()

update_lr(optimizer, 0)

optimizer.zero_grad()
loss_fn(model(x), 1).backward()
optimizer.step()

for p in model.parameters():
    print(p.data)
print(model.bn.running_mean)
print()


model.apply(fix_norm)

for p in model.parameters():
    print(p.data)
print(model.bn.running_mean)
print()



