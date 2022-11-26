import click
import pandas as pd

from sklearn.model_selection import train_test_split


@click.command()
@click.argument('in_fpath', type=click.Path(exists=True))
@click.argument('train_fpath', type=click.Path(exists=False))
@click.argument('test_fpath', type=click.Path(exists=False))
@click.option('--test_size', type=float, default=0.1)
@click.option('--random_state', type=int, default=8954892)
def split(in_fpath, train_fpath, test_fpath, test_size, random_state):
    df = pd.read_csv(in_fpath, sep='\t')
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    train.to_csv(train_fpath, sep='\t', index=False)
    test.to_csv(test_fpath, sep='\t', index=False)


if __name__ == '__main__':
    split()
