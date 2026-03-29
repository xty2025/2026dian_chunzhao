# 实验记录

## 1. MLP + Iris

- 数据集：Iris，4 维特征，3 分类。
- 模型：单隐藏层 MLP。
- 特别处理：Softmax 与 Cross Entropy 均手工实现，仅依赖 PyTorch 基础算子。
- 预期输出：训练损失下降，测试集准确率稳定在较高水平。

## 2. Attention 机制

### 2.1 Standard MHA

- 使用 `nn.Linear` 完成 Q/K/V/O 投影。
- 使用缩放点积注意力与因果 Mask。
- 验证输入输出 shape 保持一致。

### 2.2 KV Cache

- 初始输入长度设为 10。
- 随后模拟 5 次流式生成，每次输入长度为 1。
- 观测点：Q 的序列长度始终为 1；K/V Cache 的长度由 10 增长到 15。

### 2.3 GQA / MQA / MHA 统一实现

- 通过 `num_kv_heads` 统一控制：
- `num_kv_heads = num_heads` 时为 MHA。
- `num_kv_heads = 1` 时为 MQA。
- 其他情况为 GQA。

## 3. Gated DeltaNet + Fashion-MNIST

### 3.1 核心思路

- 将 28x28 图像按“行”切为长度为 28 的序列，每个 token 维度为 28。
- 先做线性投影到模型维度，再输入多层 GDN Block。
- GDN 核心状态使用每个 head 一个 `d x d` 状态矩阵，按论文中的递推式更新。
- 分类头使用全局平均池化后接线性层输出 10 类。

### 3.2 已尝试策略

- 可学习位置编码：为视觉序列补充位置信息。
- Pre-Norm 残差结构：提升训练稳定性。
- Dropout：缓解过拟合。

### 3.3 后续可继续优化的方向

- 使用 patch token 化替代逐行 token 化。
- 进一步实现真正的 chunkwise 并行前缀扫描版本。
- 与同参数量 ViT 做系统性显存/吞吐对比。

## 4. 选做实验：GDN vs ViT

### 4.1 ViT 基线

- 新增了与 GDN 同量级配置的 ViT 分类器，沿用相同的逐行序列化输入方式。
- 本次最小验证配置为 `model_dim=96, num_heads=8, depth=3, epochs=1`。
- 参数量：
- `GDN = 393,562`
- `ViT = 361,162`

### 4.2 Fashion-MNIST 最小训练结果

- GDN 一轮结果：
- `val_acc = 0.7767`
- `test_acc = 0.7702`
- ViT 一轮结果：
- `val_acc = 0.8005`
- `test_acc = 0.7855`

### 4.3 推理显存压力测试

- 设置：`Batch Size = 1`
- 输入：随机 token 序列，`token_dim = 28`
- 序列长度：`28, 56, 84, 112, 140, 168, 196, 224`
- 测试对象：
- `gdn_parallel`
- `gdn_recurrent`
- `vit`

观测结果：

- `gdn_recurrent` 的峰值显存几乎不随序列长度增长，从 `12.08MB` 增长到 `12.32MB`。
- `gdn_parallel` 随长度缓慢增长，从 `12.15MB` 增长到 `13.03MB`。
- `vit` 增长最明显，从 `12.15MB` 增长到 `15.71MB`。

结论：

- 在当前实现下，递归式 GDN 推理显存最稳定，符合其 `O(1)` 推理状态更新的设计预期。
- ViT 的注意力矩阵开销会随序列长度增加而更快放大，因此长序列推理更容易成为显存瓶颈。
