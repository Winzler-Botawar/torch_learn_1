import torch,torchvision

model = torchvision.models.resnet18(pretrained = True)
data = torch.rand(1,3,64,64)    # size=1 RGB3 weight and hight are 64
labels = torch.rand(1,1000)

prediction = model(data)
loss = (prediction-labels).sum()    # all loss plus
loss.backward()
optimization = torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.9) # learnrate = lr
optim.step()        # random gard descent active

a = torch.tensor([2.,3.],requires_grad=True)
a = torch.tensor([6.,4.],requires_grad=True)
Q = 3*a**3-b**2
external_grad = torch.tensor([1.,1.])
Q.backward(gradient = external_grad)        # now grads down to the a and b's .grad property

print(9*a**2 == a.grad)
print(-2*b == b.grad)
