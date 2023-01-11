from cpp_base import CPPBase
import cf_c

class Engine(CPPBase):
    def __init__(self, dataset=None, aggregator_weights=None, model=None, cf_config=None):
        super().__init__()
        self.c_class = cf_c.modules.train.Engine
        #   self.dataset = dataset
        #   self.model = model
        #   self.cf_config = cf_config
        #   self.aggregator_weights = aggregator_weights

        #   self.init_c_instance(dataset=self.dataset.c_instance, aggregator_weights=self.aggregator_weights.c_instance, 
        #     model=self.model.c_instance, cf_config=self.cf_config.c_instance)

        self.init_c_instance(dataset=dataset.c_instance, aggregator_weights=aggregator_weights.c_instance, model=model.c_instance, cf_config=cf_config.c_instance)

    def train_one_epoch(self):
        loss = self.c_instance.train_one_epoch()
        return loss
