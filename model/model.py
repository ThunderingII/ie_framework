import torch
import torch.nn  as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def run_train(self, config_train, config_valid,
                  dataset_train, dataset_valid,
                  collect_fn, evaluate_fn_list):
        raise NotImplementedError('模型需要实现该方法')

    def run_predict(self,):
        raise NotImplementedError('模型需要实现该方法')
