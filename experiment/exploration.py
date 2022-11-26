import copy
import json
import torch
import random

from copy import deepcopy
from os.path import join, isfile
from typing import Dict, Any

from model.training_and_testing import train, test
from model.transformer import LearnerModel
from data_processing.loading import load_cv_splits, load_assert
from utilities.logging_utils import start_logging


def cross_search(data_dir: str,
                 output_dir: str,
                 model_space: Dict[str, Any],
                 train_space: Dict[str, Any],
                 device: str,
                 seed: int):

    logger = start_logging(output_dir)
    logger.info(f'Seed: {seed}')
    logger.info(f'Model hyperparameter space: {model_space}')
    logger.info(f'Training hyperparameter space: {train_space}')

    trans_parameters = draw_parameters(model_space)
    train_parameters = draw_parameters(train_space)
    logger.info(f'train_parameters: {train_parameters}')

    with open(join(data_dir, 'encoding_cfg.json')) as j_file:
        encoding_cfg = json.load(j_file)
    trans_parameters['vocab_size'] = len(encoding_cfg['lang']) + len(encoding_cfg['world'])
    trans_parameters['lang_len'] = encoding_cfg['lang_sent_len']
    trans_parameters['world_len'] = encoding_cfg['world_sent_len']

    logger.info(f'trans_parameters: {trans_parameters}')

    for i, (train_set, test_set) in enumerate(load_cv_splits(data_dir)):
        logger.info(f'Validation split: {i}')
        model = LearnerModel(trans_parameters)

        if device == 'parallel':
            model = torch.nn.DataParallel(model)
            device = 'cuda'
        logger.info(f'device: {device}')

        logger.info('Initialised model')

        if isfile(join(data_dir, 'assert_train.tsv')):
            logger.info('Assert pre-training')
            assert_train_set = load_assert(data_dir)
            assert_tsv_file = open(
                join(output_dir, f'assert_training_{i}.tsv'),
                'w')

            assert_train_parameters = deepcopy(train_parameters)
            # assert_train_parameters['epochs'] = 1
            model = train(assert_train_set, model,
                          tsv_file=assert_tsv_file,
                          device=device,
                          **assert_train_parameters)
            assert_tsv_file.close()

        tsv_file = open(join(output_dir, f'training_{i}.tsv'), 'w')
        trained_model = train(train_set, model,
                              tsv_file=tsv_file,
                              device=device,
                              **train_parameters)
        tsv_file.close()

        results = test(test_set, trained_model,
                       batch_size=train_parameters['batch_size'],
                       device=device)
        results.to_csv(join(output_dir, f'results_{i}.tsv'), sep='\t',
                       index=False)


def draw_parameters(space):
    return {key: random.choice(value)
            for key, value in space.items()}
