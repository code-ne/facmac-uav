import numpy as np


# 表示一种先衰减然后保持不变的调度策略（schedule为调度器）
class DecayThenFlatSchedule():

    def __init__(self,
                 start, # 初始epsilon值
                 finish,    # 最终趋于平坦阶段的epsilon值
                 time_length,   # 衰减阶段持续的时间步数（在time_length内让epsilon从start下降到finish）
                 decay="exp"):  # 衰减类型，可选"linear"或"exp"

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length  # 线性衰减每步减少的epsilon值
        self.decay = decay  # 存储衰减类型

        # 判断是否选择指数衰减，并计算相应的缩放因子
        if self.decay in ["exp"]:
            # 采用公式使得 exp 衰减在 T = time_length 时大致等于 finish：exp(-time_length / exp_scaling) = finish，所以 exp_scaling = -time_length / ln(finish)
            # 注意 finish 必须大于 0 才能计算对数，否则设为 1 避免除零错误
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    # 在时间步T处返回当前的epsilon值
    def eval(self, T):
        # 线性分支
        if self.decay in ["linear"]:
            # 计算线性衰减后的epsilon值，并确保不低于finish值（即衰减后下界为finish，进入平坦阶段）
            return max(self.finish, self.start - self.delta * T)
        # 指数分支
        elif self.decay in ["exp"]:
            # 计算指数衰减后的epsilon值，并确保不低于finish值（即衰减后下界为finish，进入平坦阶段），并用min确保不超过start值，将值限制在[finish，start]范围内
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass