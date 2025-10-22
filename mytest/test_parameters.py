import torch
import torch.nn as nn

# 创建一个简单的layer
layer = nn.Linear(10, 5)
print(f"in_features: {layer.in_features}, out_features: {layer.out_features}")

# 获取参数
params = layer.parameters()
device=next(params).device

def simple_hook( module,input, output):
    print(f"Input shape in hook: {input[0].shape}\n input is {input[0]}")
    print(f"Output shape in hook: {output.shape}\n output is {output}")

    print(f"input is {input}")
    

layer.register_forward_hook(simple_hook)

x=torch.randn(2, 10, device=device)
print(f"x is {x}")
y=layer(x)
print(f"y is {y}")

print(f"Device of params: {device}")

print(f"Type of params: {type(params)}")
print(f"Is params an iterator? {hasattr(params, '__iter__') and hasattr(params, '__next__')}")

# 尝试遍历参数
print("Parameters:")
for i, param in enumerate(params):
    print(f"Parameter {i}: shape {param.shape}, requires_grad {param.requires_grad}")

# 再次调用parameters()会得到新的生成器
params_again = layer.parameters()
print(f"params_again is params: {params_again is params}")
print(f"params_again == params: {params_again == params}")
