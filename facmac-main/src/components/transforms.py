import numpy as np
import torch as th


# 抽象基类，所有变换类都应继承自此类，定义相应接口（数据预处理规则的抽象接口）
class Transform:
    # 抽象方法，子类必须实现此方法，将输入张量进行特定变换（如何把一个tensor变换成另一个tensor）
    def transform(self, tensor):
        raise NotImplementedError
    # 抽象方法，子类必须实现此方法，推断变换后输出张量的形状shape和数据类型dtype（EpisodeBatch 在初始化时需要提前分配内存不能等到 transform 运行时才知道维度）
    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError

# One-hot 编码变换类
class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim  # 保存输出one-hot向量的维度（动作数，例如uav离散动作：6个方向）

    # 将输入张量转换为one-hot编码形式
    def transform(self, tensor):
        # tensor.shape[:-1]:保留输入张量的所有维度，除了最后一维
        # 创建一个与输入张量形状相同但最后一维为out_dim
        # 例：tensor.shape = (batch, time, agents, 1), tensor.shape[:-1] = (batch, time, agents), 则新tensor形状为：y_onehot.shape = (batch, time, agents, out_dim)
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        # 在最后一维上根据输入张量的值设置对应位置为1，其余填0，实现one-hot编码
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    # 推断输出张量的形状和数据类型(提前告诉EpisodeBatch分配内存)
    def infer_output_info(self, vshape_in, dtype_in):
        # assert vshape_in == (1,)
        return (self.out_dim,), th.float32