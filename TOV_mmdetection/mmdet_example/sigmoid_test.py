import torch
import matplotlib.pyplot as plt


x = torch.arange(-20, 80).float() / 10
plt.plot(x.numpy(), x.sigmoid().numpy(), label='sigmoid(x)')
plt.plot(x.numpy(), x.sigmoid().sigmoid().numpy(), label='sigmoid(sigmoid(x))')
plt.plot(x.numpy(), -x.sigmoid().log().numpy(), label="-log(sigmoid(x))")
print((1-torch.tensor([0, 1]).float().sigmoid()).log())
plt.legend()
plt.show()

