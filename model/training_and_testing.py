import csv
import time
import torch
from torch import nn
import pandas as pd
import logging

from typing import TextIO, Union
from torch.utils.data import DataLoader

from model.transformer import LearnerModel


def tens_to_str(t: torch.Tensor):
    return ' '.join(map(str, t.tolist()))


def train(
        dataset,
        model: Union[LearnerModel, nn.DataParallel],
        tsv_file: TextIO,
        batch_size: int,
        epochs: int,
        lr: float,
        max_grad_norm: float = 1.0,
        device: str = 'cuda'
):
    logger = logging.getLogger('contrafactives')
    logger.info(f'training device: {device}')
    logger.info(f'training lr: {lr}')
    logger.info(f'training max_grad_norm: {max_grad_norm}')

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    fieldnames = ['loss', 'epoch', 'batch_no', 'kind', 'lang_codes',
                  'mind_codes', 'world_codes', 'truth_value', 'output']
    writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr
    )
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    model.to(device)
    model.train()

    for e in range(1, epochs+1):
        start_time = time.time()
        logger.info(f'Starting epoch {e} at {start_time}')

        for i, b_data in enumerate(loader):
            b_lang, b_mind, b_world, b_kind, b_labels = map(
                lambda t: t.to(device),
                b_data
            )
            # logger.info(f'b_lang.shape: {b_lang.shape}')
            # logger.info(f'b_mind.shape: {b_mind.shape}')
            # logger.info(f'b_world.shape: {b_world.shape}')
            # logger.info(f'b_labels.shape: {b_labels.shape}')
            # logger.info(f'b_kind.shape: {b_kind.shape}')

            logits = model(b_lang, b_mind, b_world)
            b_loss = criterion(logits, b_labels.float())
            output = torch.sigmoid(logits)

            for s_loss, s_lang, s_mind, s_world, s_kind, s_label, s_output in zip(
                    b_loss.unbind(),
                    b_lang.unbind(),
                    b_mind.unbind(),
                    b_world.unbind(),
                    b_kind.unbind(),
                    b_labels.unbind(),
                    output.unbind()
            ):
                writer.writerow({
                    'loss': s_loss.item(),
                    'epoch': e,
                    'batch_no': i,
                    'kind': s_kind.item(),
                    'lang_codes': tens_to_str(s_lang),
                    'mind_codes': tens_to_str(s_mind),
                    'world_codes': tens_to_str(s_world),
                    'truth_value': s_label.item(),
                    'output': s_output.item()
                })

            optimizer.zero_grad()
            b_loss.mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        logger.info(f'End of epoch after {time.time()-start_time}')

    return model


def test(
        dataset,
        model: Union[LearnerModel, nn.DataParallel],
        batch_size: int,
        device: str = 'cuda',
        threshold: float = 0.5
):
    logger = logging.getLogger('contrafactives')
    logger.info(f'testing device: {device}')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    all_dfs = []
    with torch.no_grad():
        for i, b_data in enumerate(loader):
            b_lang, b_mind, b_world, b_kind, b_labels = map(
                lambda t: t.to(device),
                b_data
            )
            logits = model(b_lang, b_mind, b_world)
            output = torch.sigmoid(logits)
            predictions = output > threshold

            rows = []
            for s_lang, s_mind, s_world, s_kind, s_label, s_output, s_prediction in zip(
                    b_lang.unbind(),
                    b_mind.unbind(),
                    b_world.unbind(),
                    b_kind.unbind(),
                    b_labels.unbind(),
                    output.unbind(),
                    predictions.unbind()
            ):
                rows.append({
                    'batch_no': i,
                    'kind': s_kind.item(),
                    'lang_codes': tens_to_str(s_lang),
                    'mind_codes': tens_to_str(s_mind),
                    'world_codes': tens_to_str(s_world),
                    'truth_value': s_label.item(),
                    'output': s_output.item(),
                    'predictions': s_prediction.item()
                })

            all_dfs.append(pd.DataFrame(rows))
    df = pd.concat(all_dfs, ignore_index=True)
    return df
