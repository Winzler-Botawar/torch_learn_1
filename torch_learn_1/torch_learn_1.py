import torch
import numpy as np

data = [[1,2],[3,4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
x_ones = torch.ones_like(x_data)    # 继承张量
print(f"Ones Tensor: \n {x_ones} \n ")
x_rand = torch.rand_like(x_data,dtype = float) # 改写张量数据类型,格式与x_data一致，数据随机
print(f"Random Tensor: \n {x_rand} \n ")

shape = (2,3)
rand_tensor = torch.rand(shape) # 随机生成2行3列的张量
ones_tensor = torch.ones(shape) # 1张量
zero_tensor = torch.zeros(shape)# 0张量
print(f"Random Tensor: \n {rand_tensor} \n ")   
print(f"Ones Tensor: \n {ones_tensor} \n ")  
print(f"Zero Tensor: \n {zero_tensor} \n ")  
print({rand_tensor.device})
#then inject to GPU
#if torch.cuda.is_available():
#    rand_tensor = rand_tensor.to('cuda')
#then print
print({rand_tensor.device})

rand_tensor[:,1] = 0    # 张量赋值
print({rand_tensor})

t1 = torch.cat([rand_tensor,ones_tensor,zero_tensor],dim = 1)
print(f"t1 is: \n {t1},\n")       # if u dont add {},will only output "t1,"
t2 = rand_tensor.mul(rand_tensor)    #equals to rand_tensor*rand_tensor
print(f"t2 is: \n {t2} \n")

print(f"t1*t2: \n {t1.matmul(t1.T)}")     # t1 and t1's matrix multiplication

t2.add_(5)              # for all elemnets add 5 to t2 ,same to other compute like copy or divide
print(t2)

t3 = torch.ones(5)      # auto create an 5dims tensor
print(t3)
t3.add_(1)
#then 
t3 = t3.numpy()         # convert tensor to numpy type
print(f"{t3} \n")


n = np.ones(5)
np.add(n,2,out = n)     #if no "out = n" last will still be [1,1,1,1,1]
t = torch.from_numpy(n) # convert numpy to tensor type
print(f"A temsor from numpy: \n {t}")
