# `inps` 和 `input_feat`是两个**不同级别**的概念：

## **self.inps** (类级属性) 🎯
- **量化器的全局校准输入**：在`init_quant()`中初始化
- **作用**：作为量化过程中传递的**基础输入数据**，会在各模块间流动
- **形状**：`[batch_size, seq_len, hidden_dim]` 表示整个校准数据集的输入
- **更新方式**：`self.inps = self._module_forward(self.inps, layer, module_kwargs)` - 模块的输出成为下一模块的输入

## **input_feat** (局部变量) 📊
- **模块内线性层的激活特征**：在`_get_input_feat()`中捕获
- **作用**：为AWQ算法收集**每层线性层的具体输入特征**用于缩放计算
- **形状**：`{'layer_name': [batch_size, seq_len, in_features]}` 字典格式，每个键对应一个线性层
- **捕获方式**：通过钩子函数在模块前向传播时实时捕获

## **关键区别**：
```python
# inps：校准数据集的基础输入，逐层传递
self.inps → Module1 → Module2 → Module3 → ...

# input_feat：Module2内部各线性层(q_proj, k_proj等)的具体激活值
input_feat = {'attention.q_proj': tensor, 'attention.k_proj': tensor, ...}
```

**为什么需要两个**：
- `inps`驱动数据流，保证量化过程完整执行
- `input_feat`提取特征数据，为量化算法提供统计信息计算依据

这是AWQ"激活感知"不可或缺的两个层次。