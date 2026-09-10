from hyperpara import physics, L
# ------------------------------
# 1、超参数
# ------------------------------
No = physics.No    # 杂质轨道数
Nb = physics.Nb    # 浴槽格点数
t  = physics.t     # 浴槽内部跃迁
U  = physics.U     # Hubbard U
J  = physics.J     # 洪德耦合
V  = physics.V     # 杂化强度
mu = physics.mu    # 化学势
# ------------------------------
# 2、函数
# ------------------------------
def index(site, m, sigma):
    """
    根据量子数 (site, m, sigma) 计算对应的二进制位索引。
    参数：
        site  : 格点或分子轨道编号（整数，从0开始）
        m     : 轨道编号（整数，从0开始）
        sigma : 自旋（0 或 1，分别对应 ↓ 和 ↑）  
    """
    return site * No * 2 + m * 2 + sigma

def count_one(state, i):
    """
    输入一个二进制整数 state 和位索引 i，
    返回 state 中第 i 位之前（即位索引 < i）的所有位中 '1' 的个数。
    该值用于计算费米子产生/湮灭算符作用时的 Jordan-Wigner 符号：
        符号 = (-1)**{count_one(state, i)}
    参数：
        state : 整数，其二进制表示长度为 L（约定高位补0）
        i     : 位索引，范围 0 ≤ i < L
    返回：
        整数，表示前 i 位中 '1' 的总数。
    """
    # 构造掩码：提取低于 i 的位（即位 0 到 i-1）
    mask = (1 << i) - 1
    # 与 state 做按位与，得到低位部分，然后调用 bit_count()（CPU 硬件指令 POPCNT）
    return (state & mask).bit_count()

# #测试
# #假设有一个构型 state，例如二进制 101010（位5=1, 位3=1, 位1=1）
# state = 0b101010   # 十进制 42
# # 计算位索引 3 之前的 '1' 个数：位0,1,2 中 1 的个数，此处位1=1，其他为0，所以为1
# ones_before3 = count_one(state, 3)   ; ones_before4= count_one(state, 4)# 返回 
# print(f'索引3前面有{ones_before3}个1，4前面有{ones_before4}个1')


def H_trans_rules():   
    """
    构造哈密顿量的跃迁规则列表，给出所有的跃迁规则，是否禁戒由作用到具体态判断
    返回 rules 列表，每个元素为元组，格式：
        ('single', coeff, src, dst)          # 单粒子跃迁及其厄米共轭
        ('exchange', -J, srcp, dstp, src, dst) # 交换项
        ('pair', +J, src1, src2, dst1, dst2)   # 配对项
    """
    # 初始化空列表
    rules = []
    # ==================== 第一部分：单粒子跃迁 ==================== 
    for site in range(0, Nb): #Nb不是总格点数，而是浴池格点数！故不用Nb-1
        for m in range(No):
            for sigma in range(2):
                # 源位：site+1，目标位：site（向左跳）
                src = index(site + 1, m, sigma)
                dst = index(site, m, sigma)

                # 根据 site 选择系数
                if site == 0:
                    coeff = V
                else:  # site >= 1
                    coeff = t

                # 添加正项：从 src 到 dst
                rules.append(('single', coeff, src, dst))
                # 添加厄米共轭项：从 dst 到 src，系数相同（实系数）
                rules.append(('single', coeff, dst, src))

    # ==================== 第二部分：交换和配对项 ====================
    # 遍历 m 和 mp，仅当 m != mp 时操作
    for m in range(No):
        for mp in range(No):
            if m == mp:
                continue
            # ---- 交换项 (exchange) ----
            srcp = index(0, mp, 0);  dstp = index(0, mp, 1)   
            src  = index(0, m,  1);  dst  = index(0, m,  0)   
            rules.append(('exchange', -J, srcp, dstp, src, dst))

            # ---- 配对项 (pair) ----
            src1 = index(0, mp, 0);  src2 = index(0, mp, 1)   
            dst1 = index(0, m,  1);  dst2 = index(0, m,  0)   
            rules.append(('pair', +J, src1, src2, dst1, dst2))

    # 返回构造好的规则列表
    return rules

rules = H_trans_rules()

def H_act_on(s): # 哈密顿量对角项在这里定义！
    """
    计算哈密顿量 H 作用于态 |s> 的结果。
    输入：
        rules : 跃迁规则列表，由 H_trans_rules() 生成
        s     : 整数，表示一个占据数构型（位长度为 L）
    返回：
        results : dict，键为输出构型（整数），值为对应的系数（复数或实数）
    """
    
    # 初始化结果字典（可能多个规则贡献到同一个最终态，需累加）
    results = {}

    # 辅助函数：向字典添加系数（若键已存在则累加）
    def add_result(key, coeff):
        if key in results:
            results[key] += coeff
        else:
            results[key] = coeff

    # 遍历所有规则
    for rule in rules:
        typ = rule[0]   # 规则类型字符串

        # ==================== 1. 单粒子跃迁 ====================
        if typ == 'single':
            coeff, src, dst = rule[1], rule[2], rule[3]
            # 条件：src 位为 1，dst 位为 0
            if ((s >> src) & 1) == 1 and ((s >> dst) & 1) == 0:
                # 翻转 src 为 0
                s_srcflip = s & ~(1 << src)
                # 计算 src 位的费米子符号,等价于(-1)**conutone，但可避免幂运算的浮点数开销
                sgnsrc = -1 if (count_one(s, src) % 2) else 1
                # 翻转 dst 为 1
                s_final = s_srcflip | (1 << dst)
                # 计算 dst 位在新态中的符号（基于 s_srcflip）
                sgndst = -1 if (count_one(s_srcflip, dst) % 2) else 1
                # 总系数
                total_coeff = coeff * sgnsrc * sgndst
                add_result(s_final, total_coeff)

        # ==================== 2. 交换项 ====================
        elif typ == 'exchange':
            coeff, srcp, dstp, src, dst = rule[1], rule[2], rule[3], rule[4], rule[5]
            # 条件：srcp=1, dstp=0, src=1, dst=0
            if (((s >> srcp) & 1) == 1 and ((s >> dstp) & 1) == 0 and
                ((s >> src) & 1) == 1 and ((s >> dst) & 1) == 0):
                # 第1次翻转：srcp → 0
                s1 = s & ~(1 << srcp)
                sgn1 = -1 if (count_one(s, srcp) % 2) else 1
                # 第2次翻转：dstp → 1
                s2 = s1 | (1 << dstp)
                sgn2 = -1 if (count_one(s1, dstp) % 2) else 1
                # 第3次翻转：src → 0
                s3 = s2 & ~(1 << src)
                sgn3 = -1 if (count_one(s2, src) % 2) else 1
                # 第4次翻转：dst → 1
                s_final = s3 | (1 << dst)
                sgn4 = -1 if (count_one(s3, dst) % 2) else 1
                total_coeff = coeff * sgn1 * sgn2 * sgn3 * sgn4
                add_result(s_final, total_coeff)

        # ==================== 3. 配对项 ====================
        elif typ == 'pair':
            coeff, src1, src2, dst1, dst2 = rule[1], rule[2], rule[3], rule[4], rule[5]
            # 条件：src1=1, src2=1, dst1=0, dst2=0
            if (((s >> src1) & 1) == 1 and ((s >> src2) & 1) == 1 and
                ((s >> dst1) & 1) == 0 and ((s >> dst2) & 1) == 0):
                # 翻转 src1 → 0
                s1 = s & ~(1 << src1)
                sgn1 = -1 if (count_one(s, src1) % 2) else 1
                # 翻转 src2 → 0
                s2 = s1 & ~(1 << src2)
                sgn2 = -1 if (count_one(s1, src2) % 2) else 1
                # 翻转 dst1 → 1
                s3 = s2 | (1 << dst1)
                sgn3 = -1 if (count_one(s2, dst1) % 2) else 1
                # 翻转 dst2 → 1
                s_final = s3 | (1 << dst2)
                sgn4 = -1 if (count_one(s3, dst2) % 2) else 1
                total_coeff = coeff * sgn1 * sgn2 * sgn3 * sgn4
                add_result(s_final, total_coeff)

    # ==================== 4. 对角项 Hloc（不改变态） ====================
    locn = 0.0  # 使用浮点数以防系数非整数

    # 4.1 mu * n_{m,sigma} 项
    for m in range(No):
        for sigma in range(2):
            bit = (s >> index(0, m, sigma)) & 1
            locn += mu * bit

    # 4.2 U * n_{m,↑} * n_{m,↓} 项
    for m in range(No):
        n_up = (s >> index(0, m, 1)) & 1
        n_dn = (s >> index(0, m, 0)) & 1
        locn += U * n_up * n_dn

    # 4.3 m ≠ m' 的项：n_{m,0} * n_{m',1}
    for m in range(No):
        for mp in range(No):
            if m == mp:
                continue
            n_m0 = (s >> index(0, m, 0)) & 1
            n_mp1 = (s >> index(0, mp, 1)) & 1
            locn += (U-2*J)*n_m0 * n_mp1   # 系数为 1（没有系数前缀？描述直接相加）

    # 4.4 m < m' 且同自旋的项： (U-3J) * n_{m,sigma} * n_{m',sigma}
    for m in range(No):
        for mp in range(No):
            if m >= mp:
                continue
            for sigma in range(2):
                n_m = (s >> index(0, m, sigma)) & 1
                n_mp = (s >> index(0, mp, sigma)) & 1
                locn += (U - 3 * J) * n_m * n_mp

    # 将对角项加入结果（键为 s 自身）
    add_result(s, locn)

    return results


def get_conn(s):
    """
    输入一个整数 s，返回所有通过哈密顿量 H 与 s 相连的构型（邻居）列表。
    邻居定义为 H_act_on(s) 返回的字典中，除 s 自身以外的所有键。
    依赖全局变量 rules。
    """
    # 调用 H_act_on 获得所有输出构型及系数
    res_dict = H_act_on(s)
    # 取出所有键，排除 s 自身
    neighbors = [key for key in res_dict.keys() if key != s]
    return neighbors


def H_mat_element(sa, sb):
    """
    输入两个整数构型 sa 和 sb，计算矩阵元 <sa|H|sb>。
    若 sa 不在 H|sb> 的结果中，返回 0；否则返回对应的系数。
    依赖全局变量 rules。
    """
    # 计算 H|sb>
    res_dict = H_act_on(sb)
    # 检查 sa 是否在结果字典中
    if sa in res_dict:
        return res_dict[sa]
    else:
        return 0.0   # 实数系数，返回 0.0（也可返回 0）


# ==================== 转换函数 ====================
def str_to_int(bin_str):
    """
    将长度为 L 的二进制字符串转换为整数。
    约定：bin_str[0] 对应整数第 0 位（最低位），bin_str[-1] 对应最高位。
    """
    num = 0
    for i, ch in enumerate(bin_str):
        if ch == '1':
            num |= (1 << i)
    return num

def int_to_str(num, L):
    """
    将整数转换为长度为 L 的二进制字符串，位0对应字符串第0个字符。
    """
    # 从位 L-1 到 0 依次取位，但我们要从位0开始，所以构建列表再反转
    chars = ['0'] * L
    for i in range(L):
        if (num >> i) & 1:
            chars[i] = '1'
    return ''.join(chars)  # 注意：chars[0] 是位0，即字符串左端



# ==================== 测试 ====================
print("规则数量:", len(rules))

# 输入字符串"0101011011100010"
s_str = "0101011011100010"
s = str_to_int(s_str)
print(f"测试构型字符串: {s_str}")
print(f"转换后的整数: {bin(s)}")  # 此处 bin 显示的是正常二进制（高位在左）

# 调用 H_act_on
res_dict = H_act_on(s)

# 打印所有键值对，键用字符串显示（位0在左）
print("\nH|s> 的所有输出构型及其系数:")
for key, val in res_dict.items():
    # 注意：我们需要知道 L 来打印固定长度，这里 L=16
    key_str = int_to_str(key, L)
    print(f"  {key_str} : {val}")

# 单独显示等于 s 的键的值（对角项）
diag_val = res_dict.get(s)
diag_str = int_to_str(s, L)
diag_val_mat=H_mat_element(s,s)
print(f"\n对角项 <{diag_str}|H|{diag_str}> = {diag_val}={diag_val_mat}")

# 调用 get_conn
neighbors = get_conn(s)
print(f"\n邻居数量: {len(neighbors)}")
print("邻居列表 (字符串，第0位在左):")
for n in neighbors:
    print(f"  {int_to_str(n, L)}")

