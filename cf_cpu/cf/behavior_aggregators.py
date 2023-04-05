import torch
import torch.nn as nn

from cpp_base import CPPBase
import cf_c


class AggregatorWeights(CPPBase, nn.Module):
    def __init__(self, config):
        super().__init__()
        nn.Module.__init__(self)
        self.c_class = cf_c.modules.behavior_aggregators.AggregatorWeights

        self.f_c0 = nn.Linear(config.emb_dim, config.emb_dim, bias=False, dtype=torch.float32)
        nn.init.normal_(self.f_c0.weight, std=1e-2)
        self.aggregator_weights0 = self.f_c0.weight.detach().cpu().numpy()
        self.init_c_instance(aggregator_weights0=self.aggregator_weights0)
