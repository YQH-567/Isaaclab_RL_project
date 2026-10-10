import torch 


pole_vel = torch.tensor([0.1, 0.2, 0.3])
print(pole_vel)  # 输出原始张量
print(pole_vel.unsqueeze(dim=1))  # 在第 1 个维度增加一个维度，变成 shape=(3, 1)
print(pole_vel.unsqueeze(dim=0))  # 在第 0 个维度增加一个维度，变成 shape=(1, 3)
print(pole_vel.shape)  # 输出原始张量的形状
print(pole_vel.unsqueeze(dim=1).shape)
print(pole_vel.unsqueeze(dim=0).shape)