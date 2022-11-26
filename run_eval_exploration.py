import json
import click

from pathlib import Path

from evaluation.result_evaluation import eval_search


@click.command()
@click.argument('search_dir', type=click.Path(exists=True))
@click.argument('encoding_fpath', type=click.Path(exists=True))
def run_eval(search_dir, encoding_fpath):
    with open(encoding_fpath) as j_file:
        encoding_cfg = json.load(j_file)

    kind_codes = {v: k for k, v in encoding_cfg['kind_codes'].items()}
    result_df = eval_search(search_dir, kind_codes)
    save_path = Path(search_dir) / 'search_results.tsv'
    result_df.to_csv(save_path, sep='\t')


if __name__ == '__main__':
    run_eval()
