import json
import click
import random

from os.path import join, isdir
from os import makedirs

from experiment.exploration import cross_search
from utilities.repro import set_seed


@click.command()
@click.argument('settings_fpath', type=click.Path(exists=True))
def run(settings_fpath):
    with open(settings_fpath) as j_file:
        settings = json.load(j_file)

    for i in range(settings['n_draws']):
        seed = random.randint(0, 9999999)
        set_seed(seed)
        output_dir = join(settings['output_dir'], str(seed))
        if not isdir(output_dir):
            makedirs(output_dir)

        cross_search(settings['data_dir'],
                     output_dir,
                     settings['model_space'],
                     settings['train_space'],
                     settings['device'],
                     seed=seed)


if __name__ == '__main__':
    run()
