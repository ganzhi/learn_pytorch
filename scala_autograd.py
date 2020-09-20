import torch

x = torch.ones(2, 2, requires_grad=True)
print("X:", x)
print("Gradient function of X:", x.grad_fn)

y = x + 2
print("Y:", y)
print("Gradien function of Y:", y.grad_fn)


z = y * y * 3
out = z.mean()
print("Z: ", z)
print("Mean of z: ", out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

out.backward()
print("Gradient of X after backward(): ", x.grad)

x = x-x.grad
x = x.detach()
x.requires_grad_(True)
print("Gradient function of X:", x.grad_fn)
z = x + 2
z = z * z * 3
out = z.mean()
print("New mean of z: ", out)

out.backward()
print("Gradient of X after backward(): ", x.grad)

x = x-x.grad
x = x.detach()
x.requires_grad_(True)
print("Gradient function of X:", x.grad_fn)
z = x + 2
z = z * z * 3
out = z.mean()
print("New mean of z: ", out)

out.backward()
print("Gradient of X after backward(): ", x.grad)

x = x-x.grad
x = x.detach()
x.requires_grad_(True)
print("Gradient function of X:", x.grad_fn)
z = x + 2
z = z * z * 3
out = z.mean()
print("New mean of z: ", out)
