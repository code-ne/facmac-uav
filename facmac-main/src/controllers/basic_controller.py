from ..modules.agents import REGISTRY as agent_REGISTRY
from ..components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
# 共享参数的多智能体控制器
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        # Ensure action_spaces is present for continuous-action environments
        # 判断是否有 action_spaces 属性，如果没有则创建一个默认的连续动作空间
        # TODO：加这一段的作用是什么？
        if not hasattr(self.args, 'action_spaces') or self.args.action_spaces is None:
            try:
                # 使用 gym.spaces.Box 创建连续动作空间
                from gym.spaces import Box
                # 优先使用 n_actions 属性，如果没有则使用 act_dim 属性，默认值为 1
                a_dim = getattr(self.args, 'n_actions', getattr(self.args, 'act_dim', 1))
                v_max = getattr(self.args, 'v_max', 1.0)
                # 为每个智能体创建一个 Box 连续动作空间，并将其赋值给 args.action_spaces 属性
                self.args.action_spaces = [Box(low=-v_max, high=v_max, shape=(a_dim,)) for _ in range(self.n_agents)]
            except Exception:
                self.args.action_spaces = None
        input_shape = self._get_input_shape(scheme)
        # 创建agent网络
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        if args.action_selector is not None:
            self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    # ep_batch: 包含观测、动作等信息的批次数据
    # t_ep: 当前时间步
    # t_env: 当前环境时间步
    # bs: 批次切片，默认为 None，表示选择所有样本
    # test_mode: 是否为测试模式，默认为 False
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, explore=False):
        # 取出该时间步的可用动作。离散动作空间中，有些动作在某些状态下可能不可用，连续空间下，动作则通常都可用
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        # 通过前向传播获取智能体的输出，根据是否为测试模式决定是否返回 logits。测试/评估策略会影响返回的是logits还是概率分布
        # obs->agent_outputs->logits/probs->actions
        agent_outputs = self.forward(ep_batch, t_ep, return_logits=(not test_mode))
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode, explore=explore)
        if getattr(self.args, "use_ent_reg", False):
            return chosen_actions, agent_outputs
        return chosen_actions

    def forward(self, ep_batch, t, return_logits=True):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # 离散动作下的处理
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                # 确保不可用动作的logits值非常小，从而在softmax后对应的概率接近于0
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            if return_logits:
                return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    # 初始化RNN的隐藏状态
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    # 返回智能体网络的参数
    def parameters(self):
        return self.agent.parameters()

    # 返回智能体网络的命名参数
    def named_parameters(self):
        return self.agent.named_parameters()

    # 将另一个BasicMAC的状态加载到当前实例中
    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def load_state_from_state_dict(self, state_dict):
        self.agent.load_state_dict(state_dict)

    def cuda(self, device="cuda"):
        self.agent.cuda(device=device)

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def share(self):
        self.agent.share_memory()

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        # 添加上一个动作的独热编码作为输入
        if self.args.obs_last_action:
            if t == 0:
                # 初始时刻填充 0
                action_key = "actions_onehot" if "actions_onehot" in batch else "actions"
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                action_key = "actions_onehot" if "actions_onehot" in batch else "actions"
                inputs.append(batch["actions_onehot"][:, t-1])
        # 添加智能体的身份信息作为输入
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        try:
            inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        except Exception as e:
            pass
        return inputs

    def _get_input_shape(self, scheme):
        # 提前计算网络输入的总维度：input_shape = obs_dim + last_action_dim + agent_id_dim
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            # 优先寻找 actions_onehot (离散)，找不到则寻找 actions (连续)
            input_shape += scheme["actions_onehot"]["vshape"][0]
        elif "actions" in scheme:
            input_shape += scheme["actions"]["vshape"][0]
        else:
            # 最后的保底方案：直接使用参数中的动作维度
            input_shape += self.args.n_actions

        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))