from __future__ import annotations

import torch
from torch import nn


def custom_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    #为什么要keepdim=True？因为我们需要保持原始维度，以便在后续计算中正确广播。
    exp_logits = torch.exp(shifted)
    return exp_logits / exp_logits.sum(dim=dim, keepdim=True)

#logits:[B,C]。targets:[B]，每个元素是类别索引。unsqueeze(1)将targets从[B]变为[B,1]，以便在gather中正确索引。
# gather(dim=1)从probs中提取每个样本对应类别的概率值，结果是[B,1]。squeeze(1)将结果从[B,1]变回[B]，得到每个样本对应类别的概率值。最后取负对数并求平均，得到最终的交叉熵损失值。
def custom_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = custom_softmax(logits, dim=-1).clamp_min(1e-9)#最小值截到这个精度
    #target_probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)#又得到概率值
    target_probs = probs[torch.arange(probs.size(0)), targets]#等价于上面gather的操作，直接用索引方式获取每个样本对应类别的概率值
    return -torch.log(target_probs).mean()#batch取平均


class IrisMLP(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, num_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.act(self.fc1(x))
        logits = self.fc2(hidden)
        probs = custom_softmax(logits, dim=-1)
        return logits, probs

# class MLP(nn.Module):
#     def __init__(self,input_dim=input_dim,hidden_dim=hidden_dim,num_class=num_class):
#         super().__init__()
#         self.fc1=nn.Linear(input_dim,hidden_dim)
#         self.act1=nn.ReLU()
#         self.fc2=nn.Linear(hidden_dim,num_class)
#     def forward(self,x):
#         hidden=self.act1(self.fc1(x))
#         logits=self.fc2(hidden)
#         probs=custom_softmax(logits,dim=-1)
#         return logits,probs