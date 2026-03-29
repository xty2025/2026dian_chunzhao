from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn

@dataclass
class GDNState:
    state_matrix: torch.Tensor  

##torch.squeeze()删除指定维度为1的维度，
# torch.unsqueeze()在指定位置插入一个维度，
#torch.view()改变张量的形状但不改变数据，
# torch.reshape()改变张量的形状但可能会返回一个新的张量，
# torch.transpose()交换张量的两个维度，torch.permute()重新排列张量的维度。view要求输入张量是连续的，而reshape不要求。view在某些情况下可能会失败，而reshape会返回一个新的张量。transpose和permute都可以交换维度，但permute更灵活，可以同时交换多个维度。
class GatedDeltaRule(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.alpha_proj = nn.Linear(model_dim, num_heads)
        self.beta_proj = nn.Linear(model_dim, num_heads)
        self.gate_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> GDNState:
        state = torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)
        return GDNState(state_matrix=state)

    def forward(self, x: torch.Tensor, state: GDNState | None = None) -> tuple[torch.Tensor, GDNState]:
        # 输入与状态初始化
        # x: (B, T, M)，其中 B=batch_size, T=seq_len, M=model_dim
        batch_size, seq_len, _ = x.shape
        # 如果没有传入先验状态，则创建初始零矩阵状态 s: (B, H, D, D)
        if state is None:
            state = self.initial_state(batch_size, x.device, x.dtype)

        # 计算 Q, K, V 并按 heads 拆分
        # q, k, v 的形状均为 (B, T, H, D)
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # alpha, beta: 从输入投影并经 sigmoid 到 (B, T, H)，再扩展到 (B, T, H, 1, 1)
        # 用作每头的可学习标量门（可随时间、样本和头变化）
        alpha = torch.sigmoid(self.alpha_proj(x)).unsqueeze(-1).unsqueeze(-1)
        beta = torch.sigmoid(self.beta_proj(x)).unsqueeze(-1).unsqueeze(-1)

        # gate: 用于读取时对每个 head 每个维度做逐元素门控，形状 (B, T, H, D)
        gate = torch.sigmoid(self._split_heads(self.gate_proj(x)))

        # s 保存当前状态矩阵，形状 (B, H, D, D)
        s = state.state_matrix
        outputs = []

        # 构造单位矩阵 I，形状 (1,1,D,D)，以便广播到 (B,H,D,D)
        identity = torch.eye(self.head_dim, device=x.device, dtype=x.dtype).view(1, 1, self.head_dim, self.head_dim)

        # 遍历时间步：对每个 step 做低秩写入与基于 query 的读取
        for step in range(seq_len):
            # 选择当前时间步的 k,v,q，形状均为 (B, H, D)
            k_t = k[:, step]
            v_t = v[:, step]
            q_t = q[:, step]
            # alpha_t, beta_t 形状为 (B, H, 1, 1)，可直接与 (B,H,D,D) 相乘
            alpha_t = alpha[:, step]
            beta_t = beta[:, step]

            # kk_t = k_t @ k_t^T (外积)，形状 (B, H, D, D)
            # 这是一个秩 1 矩阵，用于针对 k_t 的方向做擦除或控制
            kk_t = torch.matmul(k_t.unsqueeze(-1), k_t.unsqueeze(-2))

            # vk_t = v_t @ k_t^T (外积)，形状 (B, H, D, D)
            # 这是写入项，把 k->v 的关联写入到记忆矩阵中
            vk_t = torch.matmul(v_t.unsqueeze(-1), k_t.unsqueeze(-2))

            # transition = alpha_t * (I - beta_t * kk_t)
            # 表示旧记忆的衰减与针对当前 k_t 的定向擦除（由 kk_t 控制），再乘以 alpha_t
            transition = alpha_t * (identity - beta_t * kk_t)

            # 更新状态矩阵 s:
            # s <- s @ transition + beta_t * vk_t
            # 其中 s @ transition 保留并衰减旧记忆，beta_t * vk_t 为新的写入分量
            s = torch.matmul(s, transition) + beta_t * vk_t

            # 读取：y_t = s @ q_t
            # q_t.unsqueeze(-1) -> (B,H,D,1)，结果 (B,H,D,1) -> squeeze -> (B,H,D)
            y_t = torch.matmul(s, q_t.unsqueeze(-1)).squeeze(-1)

            # 使用 gate 对读取结果按元素缩放，gate[:,step] 形状 (B,H,D)
            y_t = gate[:, step] * y_t

            # 收集每步的输出（按 head 拼接后在循环外恢复到 model_dim）
            outputs.append(y_t)

        # outputs: 列表 len=T, 每项 (B,H,D) -> stack -> (B,T,H,D) -> reshape -> (B,T,M)
        output = torch.stack(outputs, dim=1).reshape(batch_size, seq_len, self.model_dim)

        # 最后经过 dropout 与线性投影回到 model_dim
        output = self.out_proj(self.dropout(output))

        # 返回输出与新的状态封装
        return output, GDNState(state_matrix=s)


class GatedDeltaNetBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(model_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(model_dim)
        self.gdn = GatedDeltaRule(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, state: GDNState | None = None) -> tuple[torch.Tensor, GDNState]:
        # x: (B, T, M)
        # 1) 先做 LayerNorm -> 送入 GatedDeltaRule（GDN）
        #    self.norm1(x): 仍为 (B, T, M)
        #    gdn_out: 来自 GatedDeltaRule 的输出，形状 (B, T, M)
        #    state: 每个 block 的记忆状态 GDNState，形状中包含 s: (B, H, D, D)
        gdn_out, state = self.gdn(self.norm1(x), state=state)

        # 2) 残差连接：把 GDN 的输出加回原始输入（逐元素相加）
        #    这与 Transformer 的残差结构保持一致，便于梯度流和训练稳定性
        x = x + gdn_out

        # 3) MLP 段：先做 LayerNorm，再通过两层线性+GELU 中间层
        #    self.norm2(x): (B, T, M) -> mlp(...) -> (B, T, M)
        #    将 MLP 输出加入残差：x = x + MLP(norm2(x))
        #    MLP 用于逐位置的非线性变换和通道混合，补充 GDN 提供的时序记忆信息
        x = x + self.mlp(self.norm2(x))

        # 返回新的特征与当前 block 的状态（供递归推理使用）
        return x, state


class GatedDeltaNetClassifier(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        token_dim: int = 28,
        max_seq_len: int = 256,
        model_dim: int = 128,
        num_heads: int = 8,
        depth: int = 4,
        num_classes: int = 10,
        dropout: float = 0.1,
        use_positional_embedding: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.token_dim = token_dim
        self.max_seq_len = max_seq_len
        self.seq_len = image_size
        self.input_proj = nn.Linear(token_dim, model_dim)
        self.use_positional_embedding = use_positional_embedding
        if use_positional_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, model_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.blocks = nn.ModuleList(
            [
                GatedDeltaNetBlock(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_classes)

    def image_to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        return x

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, token_dim = tokens.shape
        if token_dim != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {token_dim}")
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        # 输入投影：将每个 token 从 token_dim 映射到 model_dim
        # tokens: (B, T, token_dim) -> x: (B, T, model_dim)
        x = self.input_proj(tokens)

        # 可选的位置编码：为序列位置添加可学习偏置
        # pos_embedding: (1, max_seq_len, model_dim) -> 切片后广播到 (B, T, model_dim)
        if self.use_positional_embedding:
            x = x + self.pos_embedding[:, :seq_len]

        # 堆叠 Block：逐个 block 处理整个序列（非递归模式）
        # 每个 block 返回新的 x 以及 block 的状态（这里传入 state=None，即不保留跨序列的记忆）
        # block 内部包含 GDN（基于矩阵记忆的时序操作）与 MLP 段，每个 block 都有残差连接
        for block in self.blocks:
            x, _ = block(x, state=None)

        # 最后的 LayerNorm（按位置归一化通道），然后做池化得到序列级表示
        # x: (B, T, M) -> final_norm(x): (B, T, M)
        x = self.final_norm(x)
        # 平均池化：对时间维度求均值，得到 (B, M)
        pooled = x.mean(dim=1)

        # 分类头：线性层将 (B, M) 映射到 (B, num_classes)
        return self.classifier(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(self.image_to_sequence(x))

    @torch.no_grad()
    def forward_tokens_recurrent(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, token_dim = tokens.shape
        if token_dim != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {token_dim}")
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        x = self.input_proj(tokens)
        if self.use_positional_embedding:
            x = x + self.pos_embedding[:, :seq_len]

        # 逐 block 进行递归（流式/在线）处理
        # 每个 block 单独维护自己的 state（block_state），并对序列逐步更新该 state
        for block in self.blocks:
            # 初始化该 block 的状态为 None（初始时内部会用零矩阵）
            block_state = None
            outputs = []

            # 对序列每一步做在线更新：调用 block.gdn 的前向以保持并更新内部 s 矩阵
            # 在这里我们对每个时间步传入单步的输入 x[:, step:step+1]，其形状为 (B, 1, M)
            # block.gdn 返回 token_out: (B, 1, M) 与更新后的 block_state
            for step in range(x.size(1)):
                token_out, block_state = block.gdn(block.norm1(x[:, step : step + 1]), state=block_state)

                # 将 gdn 输出与原始 token 做残差相加
                token = x[:, step : step + 1] + token_out

                # 再经过 MLP 段（基于 norm2），并加入残差：得到最终该时间步的输出 token
                token = token + block.mlp(block.norm2(token))

                # 收集每个时间步的输出（形状均为 (B,1,M)），最后按时间拼接
                outputs.append(token)

            # 将本 block 处理后的所有时间步拼接回 x，供下一个 block 使用
            # outputs 列表 -> (B, T, M)
            x = torch.cat(outputs, dim=1)

        x = self.final_norm(x)
        return self.classifier(x.mean(dim=1))

    @torch.no_grad()
    def forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens_recurrent(self.image_to_sequence(x))
