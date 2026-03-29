# 2026 算法方向春招试题

本目录按题目要求实现以下内容：

1. `MLP + Iris` 三分类任务，包含自实现 Softmax、训练与评估。
2. `Standard MHA / KV Cache / GQA` 的实现与验证脚本。
3. `Gated DeltaNet + Fashion-MNIST` 分类模型，包含训练、评估、损失与准确率可视化。
4. 选做：`ViT` 基线与 `GDN vs ViT` 推理显存压力测试。

## 目录结构

```text
src/dian_spring_test/
  data/
  models/
  utils/
scripts/
results/
```

## 环境

建议使用 Python 3.11+。

```bash
cd /root/test/dian_spring_test
pip install -r requirements.txt
```

## 运行方式

### 1. Iris 三分类

```bash
PYTHONPATH=src python scripts/train_iris.py
```

### 2. 注意力机制验证

```bash
PYTHONPATH=src python scripts/run_attention_demos.py
```

### 3. Fashion-MNIST 上训练 GDN

```bash
PYTHONPATH=src python scripts/train_fashion_mnist_gdn.py --epochs 5 --batch-size 256
```

### 4. 评估已训练 GDN

```bash
PYTHONPATH=src python scripts/evaluate_fashion_mnist_gdn.py --checkpoint ./results/gdn_best.pt --model-dim 96 --num-heads 8 --depth 3
```

### 5. 检查 GDN 并行与递归输出一致性

```bash
PYTHONPATH=src python scripts/check_gdn_recurrent_equivalence.py --model-dim 96 --num-heads 8 --depth 3
```

### 6. 训练 ViT 基线

```bash
PYTHONPATH=src python scripts/train_fashion_mnist_vit.py --epochs 5 --batch-size 256
```

### 7. 对比 GDN 与 ViT 的推理显存

```bash
PYTHONPATH=src python scripts/benchmark_memory_gdn_vs_vit.py --lengths 28 56 84 112 140 168 196 224 --model-dim 96 --num-heads 8 --depth 3
```

训练结果、曲线图、模型权重会保存到 `results/`。

## 实验说明

详见 `experiment_notes.md`。
