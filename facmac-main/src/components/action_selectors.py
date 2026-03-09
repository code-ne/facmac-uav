import torch as th
from torch.distributions import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from .epsilon_schedules import DecayThenFlatSchedule

# 作用：智能体根据网络输出的logits来选择动作。包含对动作的探索或不探索
# 有四个动作选择器：GumbelSoftmaxMultinomialActionSelector、MultinomialActionSelector、GaussianActionSelector、EpsilonGreedyActionSelector
# 每个动作选择器都有select_action方法，根据不同的策略选择动作，通过配置文件决定使用哪个动作选择器。（例如action_selector: gumbel）
# TODO: 可以对动作选择器进行扩展，添加新的动作选择策略：Gumbel + 安全约束、Gumbel + APF 引导、协同探索动作选择器、基于注意力机制的动作选择器、风险感知 action selector等


# From https://github.com/mzho7212/LICA/blob/main/src/components/action_selectors.py
class GumbelSoftmax(OneHotCategorical):

    def __init__(self, logits, probs=None, temperature=1):
        # 调用父类的构造函数，初始化logits和probs属性
        # logits给网络输出动作打分，probs给出动作的概率分布
        super(GumbelSoftmax, self).__init__(logits=logits, probs=probs)
        # 设置一个非常小的数值，防止数值计算中的除零错误
        self.eps = 1e-20
        # 设置温度参数，后续在gumbel_softmax_sample采样中用来缩放logits，温度越低，softmax的概率几乎集中在最大项上，采样结果越接近one-hot分布（硬选择）
        # temperature越高，softmax概率分布越平滑，采样结果更随机（软选择），越低越接近argmax
        self.temperature = temperature

    # 给logits添加Gumbel噪声，让logits是可控随机性而不是完全随机
    def sample_gumbel(self):
        U = self.logits.clone()
        # 从均匀分布U(0,1)中采样，与logits形状相同
        U.uniform_(0, 1)
        return -th.log(-th.log(U + self.eps) + self.eps)

    # logits加上Gumbel噪声后，经过温度缩放的softmax，得到一个近似one-hot的概率分布
    def gumbel_softmax_sample(self):
        """ Draw a sample from the Gumbel-Softmax distribution. The returned sample will be a probability distribution
        that sums to 1 across classes"""
        y = self.logits + self.sample_gumbel()
        return th.softmax( y / self.temperature, dim=-1)

    # 得到一个硬选择的one-hot向量
    def hard_gumbel_softmax_sample(self):
        y = self.gumbel_softmax_sample()
        return (th.max(y, dim=-1, keepdim=True)[0] == y).float()

    def rsample(self):
        return self.gumbel_softmax_sample()

    def sample(self):
        return self.rsample().detach()

    def hard_sample(self):
        return self.hard_gumbel_softmax_sample()


def onehot_from_logits(logits, avail_logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # 获得logits中每个样本最大值的一热表示
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    if eps == 0.0:
        # 返回上一步的argmax_acs（纯贪心）
        return argmax_acs

    # chooses between best and random actions using epsilon greedy
    # 对logits进行softmax，得到每个动作的概率分布
    agent_outs = th.nn.functional.softmax(logits, dim=-1)
    # 计算每个样本可用动作的数量
    epsilon_action_num = avail_logits.sum(dim=-1, keepdim=True).float()
    agent_outs = ((1 - eps) * agent_outs + th.ones_like(agent_outs) * eps / epsilon_action_num)
    agent_outs[avail_logits == 0] = 0.0
    picked_actions = Categorical(agent_outs).sample()
    picked_actions = th.nn.functional.one_hot(picked_actions, num_classes=logits.shape[-1]).float()
    return picked_actions


REGISTRY = {}


class GumbelSoftmaxMultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_logits, avail_logits, t_env, test_mode=False, explore=False):
        masked_policies = agent_logits.clone()

        self.epsilon = self.schedule.eval(t_env)

        # 测试+贪婪，测试不探索
        if test_mode and self.test_greedy:
            # return one-hot action
            picked_actions = (th.max(masked_policies, dim=-1, keepdim=True)[0] == masked_policies).float()
        # 训练 + 不探索，正常训练路径
        else:
            if not explore:
                picked_actions = GumbelSoftmax(logits=masked_policies).gumbel_softmax_sample()
                # 前向是hard，反向传梯度是soft
                # hard + 保留梯度
                picked_actions_hard = (th.max(picked_actions, dim=-1, keepdim=True)[0] == picked_actions).float()
                picked_actions = (picked_actions_hard - picked_actions).detach() + picked_actions
            # 贪婪探索，纯探索用
            else:
                # choose between the best and random actions using epsilon greedy
                agent_outs = th.nn.functional.softmax(masked_policies, dim=-1)
                epsilon_action_num = avail_logits.sum(dim=-1, keepdim=True).float()
                agent_outs = ((1 - self.epsilon) * agent_outs + th.ones_like(
                    agent_outs) * self.epsilon / epsilon_action_num)
                agent_outs[avail_logits == 0] = 0.0
                picked_actions = Categorical(agent_outs).sample()
                picked_actions = th.nn.functional.one_hot(picked_actions, num_classes=masked_policies.shape[-1]).float()

        return picked_actions


REGISTRY["gumbel"] = GumbelSoftmaxMultinomialActionSelector


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args
        # 创造一个epsilon衰减调度器，用于随训练步数调整探索率epsilon
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        # 初始化epsilon值
        self.epsilon = self.schedule.eval(0)
        # 读取测试时是否采用贪婪策略的配置
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            # 把masked_policies作为概率参数传给Categorical分布，然后对每个智能体采样一个动作（网络已输出概率，直接采样，不能反传动作）
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class GaussianActionSelector():

    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, mu, sigma, test_mode=False):
        # expects the following input dimensionalities:
        # mu: [b x a x u]
        # sigma: [b x a x u x u]
        # mu为3维张量
        assert mu.dim() == 3, "incorrect input dim: mu"
        assert sigma.dim() == 3, "incorrect input dim: sigma"
        sigma = sigma.view(-1, self.args.n_agents, self.args.n_actions, self.args.n_actions)

        if test_mode and self.test_greedy:
            picked_actions = mu
        else:
            dst = th.distributions.MultivariateNormal(mu.view(-1,
                                                              mu.shape[-1]),
                                                      sigma.view(-1,
                                                                 mu.shape[-1],
                                                                 mu.shape[-1]))
            try:
                picked_actions = dst.sample().view(*mu.shape)
            except Exception as e:
                a = 5
                pass
        return picked_actions


REGISTRY["gaussian"] = GaussianActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        # Was there so I used it
        if getattr(self.args, "epsilon_decay_mode", "decay_then_flat") == "decay_then_flat":
            # 用配置中的起始epsilon、结束epsilon和退火步数初始化一个DecayThenFlatSchedule
            self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
            self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        if hasattr(self, "schedule"):
            self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:,:,0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
