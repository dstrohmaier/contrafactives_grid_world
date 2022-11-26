import json
import torch

from copy import deepcopy
from os.path import join, isfile
from typing import Dict, Any

from model.training_and_testing import train, test
from model.transformer import LearnerModel
from data_processing.loading import load_train_test, load_assert
from utilities.logging_utils import start_logging


def train_test(data_dir: str,
               output_dir: str,
               trans_parameters: Dict[str, Any],
               train_parameters: Dict[str, Any],
               device: str,
               seed: int):

    logger = start_logging(output_dir)
    logger.info(f'Seed: {seed}')

    with open(join(data_dir, 'encoding_cfg.json')) as j_file:
        encoding_cfg = json.load(j_file)
    trans_parameters['vocab_size'] = len(encoding_cfg['lang']) + len(encoding_cfg['world'])
    trans_parameters['lang_len'] = encoding_cfg['lang_sent_len']
    trans_parameters['world_len'] = encoding_cfg['world_sent_len']

    logger.info(f'trans_parameters: {trans_parameters}')
    logger.info(f'train_parameters: {train_parameters}')

    model = LearnerModel(trans_parameters)

    if device == 'parallel':
        model = torch.nn.DataParallel(model)
        device = 'cuda'
    logger.info(f'device: {device}')
    logger.info('Initialised model')

    train_set, test_set = load_train_test(data_dir)

    if isfile(join(data_dir, 'assert_validation.tsv')):
        logger.info('Assert pre-training')
        assert_train_set = load_assert(data_dir, 'validation')
        assert_tsv_file = open(
            join(output_dir, f'assert_training.tsv'),
            'w')
        assert_train_parameters = deepcopy(train_parameters)
        # assert_train_parameters['epochs'] = 1
        model = train(assert_train_set, model,
                      tsv_file=assert_tsv_file,
                      device=device,
                      **assert_train_parameters)
        assert_tsv_file.close()

    tsv_file = open(join(output_dir, 'training.tsv'), 'w')
    trained_model = train(train_set, model,
                          tsv_file=tsv_file,
                          device=device,
                          **train_parameters)
    tsv_file.close()

    results = test(test_set, trained_model,
                   batch_size=train_parameters['batch_size'],
                   device=device)
    results.to_csv(join(output_dir, 'results.tsv'), sep='\t',
                   index=False)
