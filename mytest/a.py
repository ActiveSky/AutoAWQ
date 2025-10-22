a=0%1
# print(a)

b=[]
c= [i*2 for i in b if i%2==0]
d=[i*2 if i%2==0 else 0 for i in b]

e={
    "a":1,
    "b":2,
    "c":3
}
# for i,j in e:
#     print(i)

import torch
from torch import nn

HIDDEN_SIZE = 256

class MyNet:
    def __init__(self,in_feature,out_feature):
        self.up=nn.Linear(in_feature,HIDDEN_SIZE)
        self.relu=nn.ReLU()
        self.down=nn.Linear(HIDDEN_SIZE,out_feature)
    def forward(self,x):
        x=self.up(x)
        x=self.relu(x)
        x=self.down(x)
        return x
    
if __name__ == '__main__':
    net=MyNet(10,2)
    # 输出网络结构
    print(net.up.parameters())
        