import torch
import torch.nn as nn
import numpy as np

from cpp_base import CPPBase
import cf_c

class Model(CPPBase, nn.Module):
    def __init__(self, config, embedding_initializer="lambda w: nn.init.normal_(w, std=1e-4)"):
        super().__init__()
        nn.Module.__init__(self)

        self.user_embedding = nn.Embedding(config.num_users, config.emb_dim, dtype=torch.float32)
        self.item_embedding = nn.Embedding(config.num_items, config.emb_dim, dtype=torch.float32)
        nn.init.normal_(self.user_embedding.weight, std=1e-2)
        nn.init.normal_(self.item_embedding.weight, std=1e-2)
        self.user_weights = None
        self.item_weights = None


class MatrixFactorization(Model):
    def __init__(self, config):
        super().__init__(config, embedding_initializer="lambda w: nn.init.normal_(w, std=1e-4)")
        self.c_class = cf_c.modules.models.MatrixFactorization

    def init_c_instance(self, config=None):
        if self.c_class is None:
            raise RuntimeError("c_class is None. ")
        
        self.user_weights = self.user_embedding.weight.detach().cpu().numpy()
        self.item_weights = self.item_embedding.weight.detach().cpu().numpy()
        self.c_instance = self.c_class(cf_config=config.c_instance, user_weights=self.user_weights, item_weights=self.item_weights)
