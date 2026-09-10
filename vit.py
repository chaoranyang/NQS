import jax.numpy as jnp
from flax import nnx
"""
输入: M 张由 0/1 构成的二维"图片" x,形状 (M, Np, NL)
      Np = 4*No 为 patch 数(每个自旋轨道一行),NL 为每条浴链长度。
      图片从大整数构型得到: 位 i 的 0/1 → 对应 (patch, 位置) 的 0/1,这部分(费米子构型 → 图片)在外部已实现,本模块只接收图片。

前向流程:
    1. Embedding: 每个 patch(长度 NL 的 0/1 向量)经线性投影映射到 d 维
    2. Positional encoding: 加上可学习的位置编码向量 (Np, d)
    3. 通过 Nl 层 Transformer 块:
    4. 输出: 对输出的Np*NL矩阵的每个元，施加 log[cosh(·)],再求和 → log ψ(s)

注:
  * 本实现使用实数参数(输出实数 log ψ)。论文一般形式写的是复值参数,
    复数可把各权重换 complex dtype,并自行定义复数 LayerNorm/GeLU
    (通常 NQS 会把模与相位拆成两个网络,或用 |ψ| 为正的模型)。

"""

class TransformerBlock(nnx.Module):
    """
    一个 Transformer 编码块。
    Pre-LN → 多头自注意力 → 残差连接 → Pre-LN → 两层 FFN(GeLU, 隐藏维 2d) → 残差连接
    """
    def __init__(self, d: int, h: int, *, rngs: nnx.Rngs):
        #  Pre-LN(注意:归一化最后一维 d)
        self.ln1 = nnx.LayerNorm(d, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d, rngs=rngs)

        #  多头自注意力(直接调包)
        # 输入 (M, Np, d) → 输出 (M, Np, d),内部完成:
        # QKV 投影 → 拆 head → softmax(q·k^T/√dk) → 加权求和 → 输出投影
        self.attn = nnx.MultiHeadAttention(
            num_heads=h,
            in_features=d,   # 每个 head 的维度 = d/h,由模块自动处理
            rngs=rngs,
        )
        #  前馈网络:两层全连接,隐藏维度 2d,激活 GeLU 
        self.ff1 = nnx.Linear(d, 2 * d, rngs=rngs)
        self.ff2 = nnx.Linear(2 * d, d, rngs=rngs)

    def __call__(self, y):
        # y: (M, Np, d)
        # 子块 1:多头自注意力 + 残差
        y = y + self.attn(self.ln1(y), decode=False) #本文情况不用decode

        # 子块 2:前馈网络 + 残差
        y = y + self.ff2(nnx.gelu(self.ff1(self.ln2(y))))

        return y



class ViTNQS(nnx.Module):
    """
    ViT 神经量子态前向传播(输出 log ψ(s))

    流程:
        输入图片 x (M, Np, NL)
          → 线性投影嵌入到维度 d
          → + 可学习位置编码
          → 通过Nl 层 TransformerBlock
          → log[cosh(·)] 非线性
          → 对所有 (Np × d) 个特征求和 → log ψ(s)  形状为(M,)

    超参数:
        d  : 隐藏维度
        h  : 注意力头数(要求 d % h == 0)
        Nl : Transformer 层数
        Np : patch 数 = 4*No
        NL : 每条链长度 = (Nb+1)/2,满足 Np*NL = L
    """
    def __init__(self, d: int, h: int, Nl: int, Np: int, NL: int, *, rngs: nnx.Rngs):
        #  1. 嵌入层:把每个 patch(长度 NL)线性投影到维度 d 
        self.embed = nnx.Linear(NL, d, rngs=rngs)

        #  2. 可学习位置编码:形状 (Np, d) 
        # nnx.Param 包装一个纯参数(非子模块)。位置编码是独立于输入的参数,
        # 广播加到每个样本的 Np 个 patch 上。
        self.pos_embed = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), (Np, d))
        )

        #  3. Nl 层 Transformer 块(list 属性,NNX 会自动追踪每个子模块)
        self.blocks = nnx.List([TransformerBlock(d, h, rngs=rngs) for _ in range(Nl)])

    def __call__(self, x):
        # x: (M, Np, NL),0/1 图片
        #  1. 嵌入 
        y = self.embed(x)            # (M, Np, d)

        #  2. 位置编码 
        y = y + self.pos_embed       # 广播 (Np, d)

        #  3. 逐层 Transformer 
        for block in self.blocks:
            y = block(y)             # (M, Np, d)

        #  4. log[cosh(·)] 数值稳定版 + 求和 
        # log(cosh(z)) = logaddexp(z, -z) - log(2),避免 |z| 大时溢出
        logcosh = jnp.logaddexp(y, -y) - jnp.log(2.0)   # (M, Np, d)

        # 对所有 (Np × d) 个特征求和,得到标量 log ψ(s)
        logpsi = jnp.sum(logcosh, axis=(1, 2))          # (M,)
        return logpsi



