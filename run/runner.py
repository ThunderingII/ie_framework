import json
import pathlib
import collections
import pandas as pd
import copy
import torch
import torch.utils.data as tud
import threading

from model.data_center import *
from config.config_cls import *
from model.evaluator import *
from model.model import BaseModel


class ModelTrainThread(threading.Thread):
    def __init__(self, model: BaseModel, config_train, config_valid,
                 dataset_train, dataset_valid, evaluators=[]):
        super().__init__()

        self.model = model
        self.config_train = config_train
        self.config_valid = config_valid
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.evaluators = evaluators
        self.finish = False
        self.start()

    def run(self):
        device_ids = self.dataset_train.device_ids
        if len(device_ids) == 1:
            device = torch.device(f'cuda:{device_ids[0]}')
            self.model.to(device)
        elif len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=device_ids)
            device = torch.device(f'cuda')
        else:
            device = torch.device(f'cpu')
            self.model.to(device)
        self.model.run_train(self.config_train, self.config_valid,
                             self.dataset_train, self.dataset_valid,
                             self.evaluators)
        self.finish = True

    def has_finish(self):
        return self.finish


class ModelPredictThread(threading.Thread):
    def __init__(self, model, dataset, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.model = model
        self.dataset = dataset
        self.start()

    def run(self):
        self.result = self.model.predict(self.dataset, **self.kwargs)

    def get_result(self):
        self.join()
        return self.result


class Runner:

    def __init__(self, config_file):
        with open(config_file) as rf:
            config = json.load(rf)
        self.config = config
        active_exp = config['active']
        base_config = config['config']
        dataset_list = config['dataset_list']
        experiment_list = config['experiment_list']

        self.base_path_model_global = pathlib.Path(
            base_config['base_path_model'])
        self.base_path_data_global = pathlib.Path(
            base_config['base_path_data'])
        self.base_path_result_global = pathlib.Path(
            base_config['base_path_result'])

        self.cache = {'dataset_list': collections.defaultdict(),
                      'model_list': collections.defaultdict()}
        self.train_runner_list = []
        self.predict_runner_list = []

        # process dataset

        # process model

        # process experiment
        for ename in active_exp:
            self._run_experiment(ename)

    def _run_experiment(self, experiment_name):
        exp_config = None
        for ec in self.config['experiment_list']:
            if ec['name'] == experiment_name:
                exp_config = ec
                break
        if exp_config and 'train' in exp_config['action_list']:
            training_config = exp_config['training']
            train_config = TrainingConfig().from_dict(
                training_config['training'])
            valid_config = ValidConfig().from_dict(
                training_config['valid_config'])
            for model_config in training_config['models']:
                train_data = self._concat_df(train_config['dataset_list'])
                valid_data = self._concat_df(valid_config['dataset_list'])
                config_local_train = copy.copy(train_config).from_dict(
                    model_config)
                config_local_valid = copy.copy(valid_config).from_dict(
                    model_config)
                model = self._get_model(model_config['model'],
                                        model_config['tag'])
                dataset_config = DatasetConfig().from_dict(
                    config_local_train['dataset_config'])
                dataset_train = eval(config_local_train['dataset'])(
                    train_data, dataset_config)
                if config_local_valid.valid_type == 'random_split':
                    vl = int(
                        config_local_valid.valid_rate * len(dataset_train))
                    dataset_train, dataset_valid = tud.dataset.random_split(
                        dataset_train,
                        [len(dataset_train) - vl, vl])
                else:
                    dataset_valid = eval(config_local_train['dataset'])(
                        valid_data, dataset_config)
                train_runner = ModelTrainThread(model, config_local_train,
                                                config_local_valid,
                                                dataset_train, dataset_valid,
                                                )
                self.train_runner_list.append(train_runner)
        if exp_config and 'evaluate' in exp_config['action_list']:
            evaluating_config = exp_config['evaluating']
            for model_config in evaluating_config['models']:
                model = self._get_model(model_config['model'],
                                        model_config['tag'])
                state_dict_path = pathlib.Path(
                    model_config[
                        'save_path']) / f"{model_config['model']}_{model_config['tag']}.pkl"
                state_dict = torch.load(state_dict_path, map_location='cpu')
                model.load_state_dict(state_dict)
                df = self._concat_df(evaluating_config['dataset_list'])
                predict_runner = ModelPredictThread(model, df, batch_size=1)

    def _concat_df(self, dataset_list):
        key = '_'.join(dataset_list)
        if key not in self.cache['dataset_list']:
            dfs = [self._get_dataset(n) for n in dataset_list]
            df = pd.concat(dfs) if dfs else None
            self.cache['dataset_list'][key] = df
        return self.cache['dataset_list'][key]

    def _get_model(self, model_name, tag) -> BaseModel:
        key = f'{model_name}_{tag}'
        if key not in self.cache['model_list']:
            self.cache['model_list'][key] = self._model_init(model_name)
        return self.cache['model_list'][key]

    def _get_dataset(self, ds_name):
        if ds_name not in self.cache['dataset_list']:
            self.cache['dataset_list'][ds_name] = self._dataset_init(ds_name)
        return self.cache['dataset_list'][ds_name]

    def _dataset_init(self, ds_name):
        dss = self.config['dataset_list']
        data_in = None
        for config_dict in dss:
            if dss['name'] == ds_name:
                base_path = pathlib.Path(config_dict['base_path']) \
                    if 'base_path' in dss else self.base_path_data_global
                data_in = pd.read_json(base_path / dss['path'])
                for action in dss['actions']:
                    t, col_name, func = action.split('|')
                    if t == 'col':
                        data_in['text'] = map(eval(func), data_in['text'])
                    elif t == 'df':
                        data_in = eval(func)(data_in)
                    elif t == 'filter':
                        df = data_in
                        data_in = df[eval(func)]
        return data_in

    def _model_init(self, model_name):
        """
        通过反射的到模型的实例
        :param model_name:
        :return:
        """
        models = self.config['models']
        for model_config_dict in models:
            if model_config_dict['name'] == model_name:
                mc = ModelConfig().from_dict(model_config_dict)
                module = __import__(mc.python_module)
                for ms in python_module.split('.')[1:]:
                    module = getattr(module, ms)
                model_class = getattr(module, mc.class_name)
                if mc.init_type == 'config':
                    config_obj = eval(f'{mc.config_class_name}()')
                    config_obj.from_dict(mc.to_dict())
                    model = model_class(config_obj)
                    return model
        return None


if __name__ == '__main__':
    python_module = 'model.crf'
    module = __import__(python_module)
    for ms in python_module.split('.')[1:]:
        module = getattr(module, ms)
    obj_model = getattr(module, 'CRF')
    # 实例化对象
    obj = obj_model(3)
    print(obj)
