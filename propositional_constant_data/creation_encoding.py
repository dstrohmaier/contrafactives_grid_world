import json
import click
import pandas as pd


VERBS = ('assert', 'non-factive', 'factive', 'contrafactive')


@click.command()
@click.argument('data_fpath', type=click.Path(exists=True))
@click.argument('out_fpath', type=click.Path(exists=False))
def create_encoding(data_fpath, out_fpath):
    data_df = pd.read_csv(data_fpath, sep='\t')
    ps = tuple(data_df.lang_p.unique())

    encoding_cfg = {'lang': {token: i for i, token in enumerate(VERBS + ps)}}

    lang_length = len(encoding_cfg['lang'])
    encoding_cfg['world'] = {f'r_{token}': i+lang_length
                             for i, token in enumerate(ps)}

    kind_codes = {token: encoding_cfg['lang'][token] for token in VERBS}

    encoding_cfg['kind_codes'] = kind_codes
    encoding_cfg['lang_sent_len'] = 2
    encoding_cfg['world_sent_len'] = 1

    with open(out_fpath, 'w') as j_file:
        json.dump(encoding_cfg, j_file, indent=4)


if __name__ == '__main__':
    create_encoding()
