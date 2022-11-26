import json
import nltk
from nltk.parse.generate import generate

import click
import random

from csv import DictWriter

from functools import lru_cache
from dataclasses import dataclass
from itertools import permutations, product, combinations


@dataclass(frozen=True)
class CellObject:
    row: int
    col: int
    colour: str
    shape: str


def create_scenarios(positions, shapes, colours):
    for pos1, pos2, pos3 in combinations(positions, 3):
        for shape1, shape2, shape3 in permutations(shapes, 3):
            for colour1, colour2, colour3 in product(colours, repeat=3):
                co1 = CellObject(row=pos1[0], col=pos1[1],
                                 colour=colour1, shape=shape1)
                co2 = CellObject(row=pos2[0], col=pos2[1],
                                 colour=colour2, shape=shape2)
                co3 = CellObject(row=pos3[0], col=pos3[1],
                                 colour=colour3, shape=shape3)
                yield (co1, co2, co3)


@lru_cache(maxsize=256)
def get_true_sentences(cell_objects):

    sentences = []

    for (co_1, co_2) in combinations(cell_objects, 2):
        if co_1.row > co_2.row:
            sentences.extend([
                [co_1.colour, co_1.shape, 'below', co_2.colour, co_2.shape],
                [co_2.colour, co_2.shape, 'above', co_1.colour, co_1.shape]
            ])
        elif co_2.row > co_1.row:
            sentences.extend([
                [co_2.colour, co_2.shape, 'below', co_1.colour, co_1.shape],
                [co_1.colour, co_1.shape, 'above', co_2.colour, co_2.shape]
            ])

        if co_1.col > co_2.col:
            sentences.extend([
                [co_1.colour, co_1.shape, 'right', co_2.colour, co_2.shape],
                [co_2.colour, co_2.shape, 'left', co_1.colour, co_1.shape]
            ])
        elif co_2.col > co_1.col:
            sentences.extend([
                [co_2.colour, co_2.shape, 'right', co_1.colour, co_1.shape],
                [co_1.colour, co_1.shape, 'left', co_2.colour, co_2.shape]
            ])

    return sentences


# @lru_cache()
def serialize_scenario(positions, scenario):
    tokens = []

    for pos in positions:
        # tokens.extend([str(p) for p in pos])
        shape_token = 'empty'
        colour_token = 'empty'

        for cell_object in scenario:
            if cell_object.row == pos[0] and cell_object.col == pos[1]:
                shape_token = cell_object.shape
                colour_token = cell_object.colour

        else:
            tokens.append(shape_token+colour_token)
    return ' '.join(tokens)


def return_verbs(o_lang,
                 scenario,
                 mind_scenario):
    world_bool = o_lang in get_true_sentences(scenario)
    mind_bool = o_lang in get_true_sentences(mind_scenario)

    # does not require: correspondence_bool = mind_representation == serialized_scenario

    return {'factive': mind_bool and world_bool,
            'contra': mind_bool and (not world_bool),
            'believe': mind_bool}


def combine_representations(positions, scenario, mind_scenario,
                            all_sentences, sample_size_false):
    data_rows = []
    true_sentences = get_true_sentences(scenario)

    serialized_world = serialize_scenario(positions, scenario)
    serialized_mind = serialize_scenario(positions, mind_scenario)

    for ts in true_sentences:
        for verb, truth_value in return_verbs(ts, scenario, mind_scenario).items():
            data_rows.append(
                {'verb': verb,
                 'truth_value': truth_value,
                 'sentence': ' '.join([verb] + ts),
                 'serialized_world': serialized_world,
                 'serialized_mind': serialized_mind}
            )

    if scenario == mind_scenario:
        mind_sentences = []
    else:
        mind_sentences = get_true_sentences(mind_scenario)

    for ms in mind_sentences:
        if ms in true_sentences:
            continue

        for verb, truth_value in return_verbs(ms, scenario, mind_scenario).items():
            data_rows.append(
                {'verb': verb,
                 'truth_value': truth_value,
                 'sentence': ' '.join([verb] + ms),
                 'serialized_world': serialized_world,
                 'serialized_mind': serialized_mind}
            )

    false_sentences = []
    while len(false_sentences) < sample_size_false:
        fs = random.choice(all_sentences)
        if fs in false_sentences:
            continue
        elif fs in true_sentences:
            continue
        elif fs in mind_sentences:
            continue

        false_sentences.append(fs)

    for fs in false_sentences:
        for verb in ('factive', 'contra', 'believe'):
            data_rows.append(
                {'verb': verb,
                 'truth_value': False,  # can only be false given that ms are ruled out
                 'sentence': ' '.join([verb] + fs),
                 'serialized_world': serialized_world,
                 'serialized_mind': serialized_mind}
            )
    return data_rows


def load_all_sentences(grammar_path):
    grammar = nltk.data.load(grammar_path)
    return list(generate(grammar))


@click.command()
@click.argument('creation_settings_file', type=click.File('r'))
@click.argument('out_path', type=click.Path(exists=False))
def create_data(creation_settings_file, out_path):
    creation_settings = json.load(creation_settings_file)

    all_scenarios = list(create_scenarios(**creation_settings['scenario']))
    print(len(all_scenarios))

    positions = creation_settings['scenario']['positions']

    all_sentences = load_all_sentences(creation_settings['grammar_path'])
    print(len(all_sentences))

    row_count = 0

    with open(out_path, 'w', newline='') as out_file:
        fieldnames = ['verb', 'truth_value', 'sentence', 'serialized_world', 'serialized_mind']
        writer = DictWriter(out_file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for scenario in all_scenarios:
            data_rows = []

            data_rows.extend(combine_representations(positions, scenario, scenario,
                                                     all_sentences, sample_size_false=5))
            mind_scenario = random.choice(all_scenarios)
            while mind_scenario == scenario:
                mind_scenario = random.choice(all_scenarios)

            data_rows.extend(combine_representations(positions, scenario, mind_scenario,
                                                     all_sentences, sample_size_false=5))

            for r in data_rows:
                row_count += 1
                writer.writerow(r)

    print(f'row count: {row_count}')


if __name__ == '__main__':
    create_data()
