import json
import click

from os.path import join, isdir
from os import makedirs

from experiment.train_test import train_test
from utilities.repro import set_seed


@click.command()
@click.argument('settings_fpath', type=click.Path(exists=True))
@click.argument('seed', type=int)
def run(settings_fpath, seed):
    with open(settings_fpath) as j_file:
        settings = json.load(j_file)

    set_seed(seed)
    output_dir = join(settings['output_dir'])
    if not isdir(output_dir):
        makedirs(output_dir)

    train_test(settings['data_dir'],
               output_dir,
               settings['trans_parameters'],
               settings['train_parameters'],
               settings['device'],
               seed=seed)


if __name__ == '__main__':
    run()
