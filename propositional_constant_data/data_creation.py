import math
import click

from csv import DictWriter

from itertools import product


def return_verbs(lang_p, world_p, mind_p):
    world_bool = lang_p == world_p
    mind_bool = lang_p == mind_p

    return {'assert': world_bool,
            'factive': mind_bool and world_bool,
            'contrafactive': mind_bool and (not world_bool),
            'non-factive': mind_bool}


def create_rows(prop_variables):
    for triple in product(prop_variables, repeat=3):
        for verb, truth_value in return_verbs(*triple).items():
            yield {'verb': verb,
                   'lang_p': triple[0],
                   'truth_value': truth_value,
                   'sentence': ' '.join([verb, triple[0]]),
                   'serialized_world': f'r_{triple[1]}',
                   'serialized_mind': f'r_{triple[2]}'}


@click.command()
@click.argument('n_prop_var', type=int)
@click.argument('out_path', type=click.Path(exists=False))
def create_data(n_prop_var, out_path):
    digits = int(math.log10(n_prop_var))+1
    prop_variables = [f'p{str(i).zfill(digits)}' for i in range(n_prop_var)]

    with open(out_path, 'w', newline='') as out_file:
        fieldnames = ['verb', 'lang_p', 'truth_value', 'sentence',
                      'serialized_world', 'serialized_mind']
        writer = DictWriter(out_file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        for row in create_rows(prop_variables):
            writer.writerow(row)


if __name__ == '__main__':
    create_data()
