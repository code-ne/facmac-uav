# 数值统计器注册表，在需要做对比实验之类的时候可以方便地调用不同环境下的数值统计器
# TODO: 后续可以添加UAV环境的数值统计器
REGISTRY = {}

try:
    from ..envs.starcraft1 import StatsAggregator as SC1StatsAggregator
    REGISTRY["sc1"] = SC1StatsAggregator
except Exception as e:
    # starcraft1 env not available
    pass

try:
    from ..envs.starcraft2 import StatsAggregator as SC2StatsAggregator
    REGISTRY["sc2"] = SC2StatsAggregator
except Exception:
    # starcraft2 env not available
    pass
