import click
import pandas as pd


@click.command()
@click.argument('in_fpath', type=click.Path(exists=True))
@click.argument('out_fpath', type=click.Path(exists=False))
def sample_down(in_fpath, out_fpath):
    full_df = pd.read_csv(in_fpath, sep='\t')
    sample_target = full_df.groupby(['verb']).truth_value.value_counts().min()
    reduced_df = full_df.groupby(['verb', 'truth_value']).sample(sample_target)
    reduced_df.to_csv(out_fpath, sep='\t', index=False)


if __name__ == '__main__':
    sample_down()
