import random
import numpy as np
from hyperpara import physics,train,Np,NL,L

NS=train.NS
No=physics.No
Nb=physics.Nb

def ini_config():
    """
    生成 NS 个互不相同的整数，每个整数二进制长度为 L = 2*No*(Nb+1)，
    且恰好有 L//2 个位为 1（其余为 0）。返回包含这些整数的列表。
    """
    #【关于前导零】物理构型00001100，其实是1100，二进制的int最高位只能为1吗？
    #“前导零”只是显示器上给人看的，在计算机的 CPU 和内存眼里，所有整数天生就带着无数个前导零！
    #你把 00001100 当作整数存储时，它变成 12。在内存中，二进制是 ...0000000000001100（前面有无数个看不见的 0）。
    #用L来限制位数，实际有效的位范围是 [0, L-1]，（但第0位是最后一个）
    random.seed(26) 
    half = L // 2          # 需要置 1 的位数

    # 【性能优化】预生成所有位对应的掩码值 (1<<i)，避免循环内重复位移
    masks = [1 << i for i in range(L)]

    # 使用集合自动去重，保证结果互不相同
    result_set = set()

    # 持续采样，直到收集到足够数量的不同构型
    # 注意：若 NS 超过组合数 C(L, half) 会死循环，但物理上 NS 通常远小于该值
    while len(result_set) < NS:
        # 从 0~L-1 中随机选择 half 个不同位置（C 级实现，极快）
        positions = random.sample(range(L), half)

        # 根据选中的位置累加掩码，直接得到整数（Python 内置 sum 在 C 循环中执行加法）
        num = sum(masks[i] for i in positions)

        # 加入集合（重复自动忽略）
        result_set.add(num)
    # 转换为列表并返回
    return list(result_set)

# #测试
# test_subspace=ini_config()
# print(f'子空间初始化完成！包含{NS}个构型，前五个为：')
# for idx, num in enumerate(test_subspace[:5]):
#     bin_str = format(num, 'b').zfill(L)
#     ones = bin_str.count('1')  
#     print(f'十进制整数{num}')
#     print(f"构型 {idx+1}: {bin_str}  (位数={len(bin_str)}, 1的个数={ones})")

# ====================================================================
# 环节 1:encode —— 把「大 int 构型列表」编码成「(N, Np, NL) 0/1 图片」
# ====================================================================
def encode(Se_list):
    """
    将一批构型(大整数)编码成 ViT 需要的 0/1 图片。

    输入:
        Se_list : list[int],长度 N,每个 int 是一个构型
                  (二进制第 b 位 = 位索引 b 的占据数,与你的 index() 约定一致)
        No, Nb  : 轨道数、浴槽格点数

    输出:
        x : np.ndarray,形状 (N, Np, NL),dtype=float32
            其中 Np = 4*No, NL = (Nb+1)/2,满足 Np*NL = L = 2*No*(Nb+1)

    编码方式(chain geometry 下的形式化拆分):
        第 b 位对应 b = site*(2*No) + spin_orbital,
            spin_orbital = m*2 + sigma ∈ [0, 2*No)
            site ∈ [0, Nb+1)

        先重排成 (N, 2*No, Nb+1):每个自旋轨道一行,每列一个 site;
        再把每行(长度 Nb+1 = 2*NL)拆成前后两半,
        最终得到 (N, 4*No, NL):
            行 r = spin_orbital*2 + half,  half=0 为前半链, half=1 为后半链
            列 c = site 在对应半段内的位置

    注意:
        当前是「链几何(chain geometry)」,没有导带/价带之分,
        因此这里只做形状重排,不做空穴变换(1-x)。
        若以后切换到自然轨道基,只需在本函数内对「价带行」做 x -> 1-x。
    """
    M = len(Se_list)
    nbytes = (L + 7) // 8          # 每个构型占多少字节

    # ---- 1. 高效地把大 int 展平成位矩阵 (N, L) ----
    bits = np.zeros((M, L), dtype=np.uint8)
    for i, num in enumerate(Se_list):
        # 转成「低位在前」的字节串,再按 bitorder='little' 展开,
        # 这样输出的第 b 个元素恰好就是构型的第 b 位(index=b)。
        raw = num.to_bytes(nbytes, 'little')
        arr = np.frombuffer(raw, dtype=np.uint8)
        bits[i] = np.unpackbits(arr, bitorder='little')[:L]

    # ---- 2. 重排成 (N, 2*No, Nb+1): [样本, 自旋轨道, site] ----
    # 位索引 b = site*(2*No) + spin_orbital,reshape 后 site 为外层(慢),
    # spin_orbital 为内层(快),再 transpose 交换两个维度。
    arr = bits.reshape(M, Nb + 1, 2 * No).transpose(0, 2, 1)
    arr = np.ascontiguousarray(arr)          # transpose 后必须连续化才能 reshape

    # ---- 3. 每条链(长度 2*NL)拆成两半 → (N, 4*No, NL) ----
    # site = half*NL + c,把最后一维切成 (half, c),
    # 再把 (spin_orbital, half) 合并成 4*No 行。
    arr = arr.reshape(M, 2 * No, 2, NL)
    x = arr.reshape(M, 4 * No, NL).astype(np.float32)
    return x