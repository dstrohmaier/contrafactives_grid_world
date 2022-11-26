import os
import logging
from os.path import join, isdir


def start_logging(logging_directory: str,
                  logger_name: str = 'contrafactives',
                  file_name: str = 'run.log'):
    if not isdir(logging_directory):
        os.makedirs(logging_directory)

    logging_path = join(logging_directory, file_name)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(logging_path)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:'
                                  '%(levelname)s:'
                                  '%(filename)s: '
                                  '%(message)s')
    fh.setFormatter(formatter)

    for old_fh in logger.handlers:  # remove all old handlers
        logger.removeHandler(old_fh)
    logger.addHandler(fh)      # set the new handler

    logger.info('Started running')
    return logger


def log_model_device(model):
    logger = logging.getLogger('sdm')

    for name, parameter in model.named_parameters():
        logger.debug(f'Model parameter "{name}" on device: {parameter.device}')
