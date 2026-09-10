"""
main_training.py —— 主循环骨架(子空间扩展 + ViT 前向 + 截断)

当前版本 SR 参数更新为占位,后续在标记处接入。
数据流:
    S --扩展--> Se --定序/建索引--> 编码成图片 --前向--> logψ --截断--> S_next
"""

import numpy as np
import jax.numpy as jnp
from flax import nnx

# ============ 导入项目各模块 ============
from hyperpara import physics, net, train, L, Np, NL
from config import ini_config, encode
from hamiltonian import H_act_on
from vit import ViTNQS

No = physics.No
Nb = physics.Nb
NS = train.NS


def run_training():
    """子空间展开 + 截断的主循环(不含 SR 更新)。"""

    # ---------------- 0. 初始化子空间与 ViT 模型 ----------------
    # 初始子空间:NS 个半满构型(大整数列表)
    S = ini_config()

    # ViT 模型:NNX 在构造函数里 eager 创建全部参数
    rngs = nnx.Rngs(0)
    model = ViTNQS(
        d=net.d,
        h=net.h,
        Nl=net.Nl,
        Np=Np,
        NL=NL,
        rngs=rngs,
    )

    print(f"系统规模: L={L}, Np={Np}, NL={NL}")
    print(f"子空间大小 NS={NS}, 总迭代轮数={train.max_iter}\n")

    # ---------------- 主循环 ----------------
    for it in range(train.max_iter):

        # ---------- (1) 扩展:收集 S 的一跳邻域 ----------
        # Se = S ∪ {所有 s∈S 的邻居}
        # 这一步只关心"键"(构型本身),不存矩阵元;
        # 矩阵元在 SR 阶段对扩展前的 S 临时计算。
        Se_set = set(S)
        for s in S:
            # H_act_on(rules, s) 返回 {s' : <s'|H|s>}
            # 其键集合正是 s 的所有邻居(含 s 自身的对角项)
            Se_set.update(H_act_on(s).keys())

        # ---------- (2) 定序 + 建索引 ----------
        # Se_list 是本轮唯一稳定索引:Se_list[i] = 第 i 个构型
        # Se_idx  提供反查:大 int -> 下标
        Se_list = sorted(Se_set)
        Se_idx = {v: i for i, v in enumerate(Se_list)}

        # ---------- (3) 编码成图片 + 一次性前向 ----------
        # 把整个扩展空间编码成 (N_se, Np, NL) 的 0/1 图片
        bits_se = encode(Se_list)              # 若encode 仍需要参数,用:
                                               # bits_se = encode(Se_list, No, Nb)
        x_jax = jnp.asarray(bits_se)           # numpy -> jax 数组
        logpsi = model(x_jax)                  # (N_se,),logψ(s)
        logpsi = logpsi - jnp.max(logpsi)      # 数值稳定,不影响 |ψ|² 的相对大小

        # ---------- (3.5) 占位:SR 参数更新(后续实现) ----------
        # 未来这里做:
        #   (a) 对"扩展前的 S"中每个构型 s 计算 E_loc(s):
        #         E_loc(s) = Σ_{s'} <s|H|s'> * exp(logψ(s') - logψ(s))
        #       其中 s' 用 Se_idx 反查下标,取 logpsi[j];
        #       s 的所有邻居都在 Se 里,因此求和精确闭合。
        #   (b) 用 vmap 对 S 中构型求 O_k(s)=∂logψ/∂θ_k。
        #   (c) 组装 S_kk'、f_k,解 (S+λI)x=f,更新 model 参数。
        # 注意:求和域是"扩展前的 S",不是截断后的 S_next。
        # ======================================================

        # ---------- (4) 截断:按 |ψ|² 取前 NS 个,生成下一轮 S ----------
        # 实数正定波函数:|ψ(s)|² = exp(2 * logψ(s))
        prob = np.asarray(jnp.exp(2.0 * logpsi))       # (N_se,)
        top_idx = np.argsort(-prob)[:NS]               # 概率从大到小,取前 NS 个下标
        S_next = [Se_list[i] for i in top_idx]         # 下标 -> 大 int 构型

        # ---------- 本轮监控 ----------
        overlap = len(set(S) & set(S_next))            # 有多少旧构型被保留
        max_prob = float(prob[top_idx[0]])
        print(
            f"iter {it:4d}: "
            f"|Se|={len(Se_list):6d}  "
            f"保留旧构型={overlap}/{NS}  "
            f"最大概率={max_prob:.4e}"
        )

        # ---------- 进入下一轮 ----------
        S = S_next

    return S, model


if __name__ == "__main__":
    S_final, model = run_training()
    print(f"\n训练骨架运行完成,最终子空间包含 {len(S_final)} 个构型。")