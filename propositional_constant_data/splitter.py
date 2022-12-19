import click
import pandas as pd

from os.path import join
from sklearn.model_selection import train_test_split


def resample(df, random_state):
    all_dfs = []

    for _, verb_df in df.groupby('verb'):
        false_df = verb_df[~verb_df.truth_value]
        true_df = verb_df[verb_df.truth_value]

        false_len = false_df.shape[0]
        true_len = true_df.shape[0]

        if false_len > true_len:
            to_extend_df = true_df
            other_df = false_df
        elif true_len > false_len:
            to_extend_df = false_df
            other_df = true_df
        else:
            all_dfs.append(false_df)
            all_dfs.append(true_df)
            continue

        extension_len = max(false_len, true_len) - min(false_len, true_len)
        extension_df = to_extend_df.sample(extension_len,
                                           replace=True)

        all_dfs.extend([to_extend_df, extension_df, other_df])

    return pd.concat(all_dfs).sample(
        frac=1,
        random_state=random_state
    ).reset_index()


@click.command()
@click.argument('in_fpath', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--validation_size', type=float, default=0.3)
@click.option('--random_state', type=int, default=1848)
def split(in_fpath, output_dir, validation_size, random_state):
    df = pd.read_csv(in_fpath, sep='\t')

    assert_df = df[df.verb == 'assert']
    main_df = df[df.verb != 'assert'].sample(
        frac=1,
        random_state=random_state).reset_index()

    train_ps, validation_ps = train_test_split(df.lang_p.unique(),
                                               test_size=validation_size,
                                               random_state=random_state)

    train_rs = {f'r_{p}' for p in train_ps}
    validation_rs = {f'r_{p}' for p in validation_ps}

    train_df = main_df[(main_df.lang_p.isin(train_ps) &
                        main_df.serialized_world.isin(train_rs) &
                        main_df.serialized_mind.isin(train_rs))]
    validation_df = main_df[(main_df.lang_p.isin(validation_ps) |
                             main_df.serialized_world.isin(validation_rs) |
                             main_df.serialized_mind.isin(validation_rs))]

    train_assert_df = assert_df[(assert_df.lang_p.isin(train_ps) &
                                 assert_df.serialized_world.isin(train_rs) &
                                 assert_df.serialized_mind.isin(train_rs))]

    train_df = resample(train_df, random_state)
    train_assert_df = resample(train_assert_df, random_state)
    assert_df = resample(assert_df, random_state)

    train_fpath, validation_fpath = (join(output_dir, f'{split}.tsv')
                                     for split in ('train', 'validation'))
    train_assert_fpath, validation_assert_fpath = (
        join(output_dir, f'assert_{split}.tsv')
        for split in ('train', 'validation')
    )
    train_df.to_csv(train_fpath, sep='\t', index=False)
    validation_df.to_csv(validation_fpath, sep='\t', index=False)
    train_assert_df.to_csv(train_assert_fpath, sep='\t', index=False)
    assert_df.to_csv(validation_assert_fpath, sep='\t', index=False)


if __name__ == '__main__':
    split()

# python splitter.py all_small_data.tsv train.tsv validation.tsv
