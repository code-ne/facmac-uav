### 项目运行命令行
1. 列出已有的 episode_*.json 文件：
``` 
dir results\**\episode_*.json /b
```
2. 运行绘图脚本：
```
python plot_rewards.py --dir results --algos maddpg,facmac
```
3. 运行项目（推荐：按 episode 数停止） -m src.-config

- 推荐用法（在命令行直接指定要跑的 episode 数，优先级高于 YAML）：
```
python -m src.main --config=facmac_uav --env-config=uav_env with test_nepisode=10000
```

- 如果你已经在算法配置文件（例如 `src/config/algs/facmac_pp.yaml`）中写入了 `t_max_episodes: 10000`，可以直接运行而不用在命令行传参：
```
python -m src.main --config=facmac_uav --env-config=uav_env
```

4. 在断点重新启用模型训练
```
python -m src.main --config=facmac_uav --env-config=uav_env with checkpoint_path="results/models/facmac_uav_navigation__2026-03-10_22-40-57"
```

说明：
- `t_max_episodes`（新增参数）—— 当存在时，训练会在累计到该数量的 episode 后停止（这是按“集数”停止训练的直观方式）。
- `t_max`（历史/旧参数）—— 表示训练要执行的总环境步数（timesteps）。如果你把 `t_max` 设为 200，而每集 `max_steps` 为 200，那么训练只会产出约 1 个 episode（200 timesteps / 200 steps-per-episode）。因此 `t_max` 并不是“每集最大步数”的设置，而是训练的总步数上限。
- 环境的单集最大步数由环境配置控制（例如 `src/config/envs/uav_env.yaml` 中的 `max_steps` 或 `episode_limit`），通常这里是你想要的“每集最大步数”（例如 200）。

4. 运行评估/测试（控制每次评估跑多少集）
```
# test_nepisode 控制一次评估/测试要跑多少个 episode（不改变训练什么时候停止）
python -m src.main --config=facmac_pp --env-config=uav_env with test_nepisode=10
```

5. 打开生成的图：
```
start "" "results\plots\mean_episode_reward.png"
start "" "results\plots\combined.png"
```
6. 列出可用算法/环境配置文件名：
```
dir src\config\algs\* -Name
dir src\config\envs\* -Name
```
7. 服务器上运行tensorboard
```angular2html
首先在服务器终端进入项目目录，然后在命令行输入：（results/logs是指tensorboard文件存放路径）
tensorboard --logdir results/logs --port 6006
然后在本地终端进行端口转发：
ssh -L 6006:localhost:6006 user@server_address（ ssh -L 6006:localhost:6006 gy2024@172.23.206.107）
最好在本地浏览器访问 http://localhost:6006 就可以看到tensorboard界面了
运行期间不要关闭服务器上的tensorboard进程，否则本地访问会断开
```

额外提示：
- 启动时在控制台会打印出使用的停止条件，示例：
  - "Beginning training for 10000 episodes" —— 表示使用 `t_max_episodes`。
  - "Beginning training for 2000000 timesteps" —— 表示使用 `t_max`（timesteps）。
- 如果你希望精确得到 N 个 episode（并为每集保存 JSON），优先使用 `t_max_episodes` 来控制训练长度。

### 项目修改建议
环境 (multi_uav_env.py):
- 障碍与目标采样 _init_world：可调 margin、障碍尺寸范围、goal 采样失败策略 (max_attempts)。可以加入 deterministic 障碍预定义帮助复现。
- 到达检测：既判断距离 <= goal_radius 又判断线段是否穿过球体 _segment_intersects_sphere。调试可在该函数加详细打印或事件列表以确认“穿越” vs “结束”。
- 碰撞检测 _collides_any：当前简单地遍历所有障碍 + 动态物体；可加缓存或空间划分加速（调试慢步性能）。
- 动作缩放与速度裁剪：a = np.array(actions[i]) * v_max 后再限幅；可调试例如采用平滑加速度模型（速度积分 vs 直接设定速度）。
- Episode summary 写文件 _finalize_episode：文件名组合 (time_ns + pid + counter)。可添加自定义 run_token 前缀或减少IO (批量写)。
- 指标定义:
1. success_rate: 当前统计为到达的 agent 数除以 n_agents。可扩展为 episode 是否全到达 (binary) 或平均到达时间指标。
2. collision_rate: 用第一次碰撞步是否发生的标志，未统计碰撞频次；可替换成 sum(collisions_step)/n_agents/steps。 
3. avg_latency_rate: 用 first_arrival_step 填充未到达为 max_steps 后再平均除以 max_steps；可改为“仅到达者平均”与“未到达者单独统计”。 奖励 (reward.py):
- 系数: ARRIVAL_BONUS, COLLISION_PENALTY, CONTROL_PENALTY_COEF, SHAPING_SCALE。可做超参网格搜索或自适应调权。
- shaping 当前用距离差，若出现抖动可增加死区或将 delta clip。
- 可增加团队奖励成分 (如全部到达 bonus)，或潜在碰撞累计惩罚。 UAV 封装 (uav_env.py):
- get_avail_actions 返回全 1（连续动作始终可用），若需要限制(如速度不同维度约束)可改为区间掩码。
- step 的 fallback 分支：如果底层环境返回格式改变，可能静默吞异常；调试可在 except 中打印 res 类型。 动作选择 (action_selectors.py):
- epsilon 与 gumbel 流程：调试探索策略在连续环境中的影响（当前 FACMAC 连续版本使用 Gaussian? 如果 MAC 选择器设为 "gaussian"）。
- 可加入对动作分布统计日志（均值/方差） -> EpisodeDualLogger。 训练主循环 (run.py):
- 停止条件: 支持 t_max 或 t_max_episodes，调试中可添加“达到收敛阈值即提前停止”逻辑。
- Buffer warmup, batch_size, learn interval（如 transition runner）。
- 目标网络更新模式：目前支持 hard/soft/EMA；可调试 tau 值与更新频率。 Learner (facmac_learner.py):
- Actor 损失 pg_loss = -chosen_action_qvals.mean() + (pi**2).mean() * 1e-3：第二项是一个 L2 正则化。可调系数或替换为熵正则。
- Critic TD 目标: targets = rewards + gamma*(1-terminated)*target_vals。可扩展 n-step 或 TD(λ)（离散版本才用 TD λ）。
- 若使用混合器(mixer)，当前对连续动作的 Q 单输出再混合；可扩展集中 Critic 接收 global state + joint action 更丰富输入。
- 反向传播中使用 mask.sum() 作为归一化；可在出现极少有效步长时加入最小 mask 防止除 0。 并行 Runner (parallel_runner.py):
- 动作范数与均值统计 (action_norms, action_means)：可拓展记录每维最大/最小动作值、分布偏度。
- 噪声重置: self.mac.ou_noise_state = actions.clone().zero_() — 若启用 OU 噪声策略可检查是否合理。 Episode Runner:
- 轨迹记录: 仅在有 env.pos 情况下采集；可加入高度随时间变化曲线或速度热图。
- _safe_add 统计融合：遇到复杂结构(列表/np数组)做转换，可产生较大 JSON；可限制键或保留精简版。 日志 (episode_logger.py):
- 目前 JSONL 每行包含所有 details；调试大规模训练时可裁剪 (如每 N 轮保存轨迹)。
- 图片写入：在 Windows 下偶尔卡顿，可以加 try/except 已处理；可扩展 GIF 生成轨迹动画。
<hr></hr>

### 潜在隐藏问题与风险点
- Arrival/collision 统计偏差：线段穿过球体的检测可能在高速时判定到达但位置尚未落入目标区域；可在日志中同时存储“进入球体时的绝对时间/距离”做校验。
- collision_rate 使用是否发生过碰撞 (first_collision_step>=0)，忽略持续碰撞次数和时长。可能无法反映安全性能。
- 奖励 shaping 与 CONTROL_PENALTY 可能在速度接近 0 时梯度很弱（norm_speed 小），导致策略不愿移动；需观察奖励范围分布。
- 多进程并行写文件 _finalize_episode：IO 瓶颈或文件系统延迟可能影响高频 episode 结束；可考虑批缓存。
- EpisodeDualLogger 图像生成: 每轮都创建 matplotlib figure 在长时间训练会内存警告；可按间隔写图。
- Continuous scheme: actions_vshape 推导逻辑 里 Box 与 Tuple 情况混合，但 UAV 使用 Box；如将 action_space shape 改变会影响 scheme 维度，需谨慎。