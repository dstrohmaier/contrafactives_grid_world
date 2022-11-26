import json
import click


LANG_VOCAB = ('factive', 'contra', 'believe',
              'circle', 'triangle', 'square',
              'red', 'blue', 'yellow',
              'above', 'below', 'left', 'right')


WORLD_VOCAB = ('circlered', 'circleblue', 'circleyellow',
               'trianglered', 'triangleblue', 'triangleyellow',
               'squarered', 'squareblue', 'squareyellow',
               'emptyempty')


VERBS = ('believe', 'factive', 'contra')


@click.command()
@click.argument('out_fpath', type=click.Path(exists=False))
def create_encoding(out_fpath):
    encoding_cfg = {'lang': {token: i for i, token in enumerate(LANG_VOCAB)}}
    encoding_cfg['world'] = {token: i+len(encoding_cfg['lang'])
                             for i, token in enumerate(WORLD_VOCAB)}

    kind_codes = {token: encoding_cfg['lang'][token] for token in VERBS}

    encoding_cfg['kind_codes'] = kind_codes
    encoding_cfg['lang_sent_len'] = 6
    encoding_cfg['world_sent_len'] = 9

    with open(out_fpath, 'w') as j_file:
        json.dump(encoding_cfg, j_file, indent=4)


if __name__ == '__main__':
    create_encoding()
