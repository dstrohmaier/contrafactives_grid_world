import click
import numpy as np
import pandas as pd

from os.path import join


@click.command()
@click.argument('in_fpath', type=click.Path(exists=True))
@click.argument('out_dir', type=click.Path(exists=True))
@click.option('-n', type=int, default=5)
def create_cv_files(in_fpath, out_dir, n):
    df = pd.read_csv(in_fpath, sep='\t')
    df = df.sample(frac=1)

    new_dfs = np.array_split(df, n)
    for i, cv_df in enumerate(new_dfs):
        out_fpath = join(out_dir, f'{i}_split.tsv')
        cv_df.to_csv(out_fpath, sep='\t', index=False)


if __name__ == '__main__':
    create_cv_files()
