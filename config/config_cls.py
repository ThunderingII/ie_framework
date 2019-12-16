def dfs_parse_config(config: dict):
    ans = []
    for key in config:
        if isinstance(config[key], dict):
            ans.extend(dfs_parse_config(config[key]))
        else:
            ans.append((key, config[key]))
    return ans


class BaseConfig:
    def __init__(self):
        self.delete_key = ['delete_key']

    def from_dict(self, config: dict):
        values = dfs_parse_config(config)
        for k, v in values:
            setattr(self, k, v)
        return self

    def __str__(self):
        d = self.__dict__
        return str(self.__class__) + '\t' + '\t'.join(
            [f'{k}:{d[k]}' for k in d])

    def to_dict(self, delete=True):
        d = self.__dict__
        if delete:
            for k in self.delete_key:
                del d[k]
        return d


class ModelConfig(BaseConfig):
    def __init__(self, python_module=None, class_name=None, name=None,
                 init_type=None, save_path=None, tag=None,
                 config_class_name=None):
        super().__init__()
        self.python_module = python_module
        self.class_name = class_name
        self.name = name
        self.tag = tag
        self.init_type = init_type
        self.save_path = save_path
        self.config_class_name = config_class_name
        self.delete_key = ['delete_key', 'python_module', 'class_name', 'name',
                           'init_type']


class TrainingConfig(BaseConfig):

    def __init__(self, lr=1e-5, batch_size=64, train_dataset=[],
                 dataset='NerDataset',
                 collect_fn='', monitor='f1', save_path='', model='', tag=''):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.dataset = dataset
        self.collect_fn = collect_fn
        self.monitor = monitor
        self.save_path = save_path
        self.model = model
        self.tag = tag


class ValidConfig(BaseConfig):
    def __init__(self, valid_type='random_split', valid_dataset=[],
                 valid_rate=0.15, model='', tag=''):
        super().__init__()
        self.valid_type = valid_type
        self.valid_dataset = valid_dataset
        self.valid_rate = valid_rate
        self.model = model
        self.tag = tag


class EvaluatorConfig(BaseConfig):
    def __init__(self, name, method_name, plot_type, extra):
        super().__init__()
        self.name = name
        self.method_name = method_name
        self.plot_type = plot_type
        self.extra = extra


class DatasetConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        pass


class NerModelConfig(ModelConfig):

    def __init__(self, labelling_type='crf', num_tags=3, mix_type='add',
                 em_type='bert', freeze_embedding_bert=True,
                 vocab_size=20000, glove_size=300, feature_size=32,
                 hidden_size=1000, dropout=0.2, bert_size=768,
                 bert_layers=12, bert_attn_heads=8, tf_size=1000,
                 tf_head_num=1000, tf_dropout=1000,
                 extra_feats_type='lstm'):
        super().__init__()
        self.dropout = dropout
        # embedding layer
        self.mix_type = mix_type
        self.em_type = em_type
        self.vocab_size = vocab_size
        self.glove_size = glove_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size

        # bert
        self.freeze_embedding_bert = freeze_embedding_bert
        self.bert_size = bert_size
        self.bert_layers = bert_layers
        self.bert_attn_heads = bert_attn_heads

        # feats extra
        self.tf_size = tf_size
        self.tf_head_num = tf_head_num
        self.tf_dropout = tf_dropout
        self.extra_feats_type = extra_feats_type

        # labeling
        self.labelling_type = labelling_type
        self.num_tags = num_tags


if __name__ == '__main__':
    config = {'lr': 0.0001, 'batch_size': 34}
    print(TrainingConfig().from_dict(config))

    d = {
        "labelling_type": "crf",
        "num_tags": 3,
        "mix_type": "add",
        "em_type": "bert",
        "freeze_embedding_bert": True,
        "vocab_size": 20000,
        "glove_size": 300,
        "feature_size": 32,
        "hidden_size": 1000,
        "dropout": 0.2,
        "bert_size": 768,
        "bert_layers": 12,
        "bert_attn_heads": 8,
        "tf_size": 1000,
        "tf_head_num": 1000,
        "tf_dropout": 1000,
        "extra_feats_type": "lstm"
    }
    t = NerModelConfig().from_dict(d)
    print(t)
    print(t.to_dict())
    # print(','.join([f'{k}={d[k]}' for k in d]))
    # print('\n'.join([f'self.{k} = {k}' for k in d]))
