import glob
import pandas as pd

from typing import List
from os.path import join
from pathlib import Path
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             brier_score_loss,
                             confusion_matrix)


METRICS = ('accuracy', 'f1', 'brier_score_loss',
           'tp', 'fp', 'fn', 'tn')


def extract_metrics(df: pd.DataFrame):
    y_true = df.truth_value
    y_pred = df.output
    y_pred_class = df.predictions

    cm = confusion_matrix(y_true, y_pred_class, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_class),
        'f1': f1_score(y_true, y_pred_class),
        'brier_score_loss': brier_score_loss(y_true, y_pred),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

    return metrics


def eval_result_df(df: pd.DataFrame, kind_codes) -> pd.DataFrame:
    rows = []

    all_row = extract_metrics(df)
    all_row['kind'] = 'all'
    rows.append(all_row)

    for kind, kind_df in df.groupby(['kind']):
        kind_row = extract_metrics(kind_df)
        kind_row['kind'] = kind_codes[kind]
        rows.append(kind_row)

    return pd.DataFrame(rows)


def eval_across_folds(fold_dfs: List[pd.DataFrame], kind_codes) -> pd.DataFrame:
    df_concat = pd.concat([eval_result_df(df, kind_codes) for df in fold_dfs])
    mean_metric_df = df_concat.groupby(['kind']).mean()
    mean_metric_df.reset_index(inplace=True)
    return mean_metric_df


def eval_search(search_dir, kind_codes) -> pd.DataFrame:
    all_dfs = []

    for seed_path in glob.glob(join(search_dir, '*/')):
        seed = Path(seed_path).stem

        fold_dfs = []
        for fpath in glob.glob(join(seed_path, 'results_*.tsv')):
            fold_dfs.append(
                pd.read_csv(fpath, sep='\t')
            )

        if fold_dfs == []:
            continue
        seed_df = eval_across_folds(fold_dfs, kind_codes)
        seed_df['seed'] = seed
        all_dfs.append(seed_df)

    return pd.concat(all_dfs).reset_index(drop=True)
