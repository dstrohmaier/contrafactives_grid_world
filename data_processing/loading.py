import glob
import json
import torch
import pandas as pd

from os.path import join
from torch.utils.data import TensorDataset, ConcatDataset


def encode(string, encoding):
    converted = [encoding[token] for token in string.split()]
    return torch.LongTensor(converted)


def df_to_dataset(df, encoding_cfg):
    lang_codes, mind_codes, world_codes, kind_codes, truth_values = [], [], [], [], []

    for _, row in df.iterrows():
        lang_codes.append(encode(row.sentence, encoding_cfg['lang']))
        mind_codes.append(encode(row.serialized_mind, encoding_cfg['world']))
        world_codes.append(encode(row.serialized_world, encoding_cfg['world']))
        kind_codes.append(encoding_cfg['kind_codes'][row.verb])
        truth_values.append(int(row.truth_value))

    dataset = TensorDataset(
        torch.stack(lang_codes),
        torch.stack(mind_codes),
        torch.stack(world_codes),
        torch.LongTensor(kind_codes).reshape(-1, 1),
        torch.LongTensor(truth_values).reshape(-1, 1)
    )
    return dataset


def load_cv_splits(data_dir: str):
    with open(join(data_dir, 'encoding_cfg.json')) as j_file:
        encoding_cfg = json.load(j_file)

    dfs = []
    for split_fpath in glob.glob(join(data_dir, 'cv_splits', '*_split.tsv')):
        dfs.append(pd.read_csv(split_fpath, sep='\t'))

    datasets = [df_to_dataset(df, encoding_cfg) for df in dfs]

    for i in range(len(datasets)):
        train_dataset = ConcatDataset(datasets[:i] + datasets[i+1:])
        test_dataset = datasets[i]

        yield train_dataset, test_dataset


def load_train_test(data_dir: str):
    with open(join(data_dir, 'encoding_cfg.json')) as j_file:
        encoding_cfg = json.load(j_file)

    train_dataset = load_tsv(join(data_dir, 'train.tsv'), encoding_cfg)
    test_dataset = load_tsv(join(data_dir, 'validation.tsv'), encoding_cfg)

    return train_dataset, test_dataset


def load_assert(data_dir, split: str = 'train'):
    with open(join(data_dir, 'encoding_cfg.json')) as j_file:
        encoding_cfg = json.load(j_file)

    assert_dataset = load_tsv(join(data_dir, f'assert_{split}.tsv'),
                              encoding_cfg)
    return assert_dataset


def load_tsv(file_path, encoding_cfg):
    df = pd.read_csv(join(file_path), sep='\t')
    dataset = df_to_dataset(df, encoding_cfg)
    return dataset
