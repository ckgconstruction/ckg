import os
import torch
import torch.nn as nn
import utils.bio as bio
from extract import extract
from utils import utils as utils
from test import do_eval
from tqdm import tqdm
import logging

logger = logging.getLogger('root')


def train(args, epoch, model, trn_loader, dev_loaders, summarizer, optimizer, scheduler):
    total_pred_loss, total_arg_loss, trn_results = 0, 0, None
    epoch_steps = int(args.total_steps / args.epochs)

    args.summary_step = int(len(trn_loader) / 3)

    iterator = tqdm(enumerate(trn_loader), desc='steps', total=epoch_steps)
    for step, batch in iterator:
        batch = map(lambda x: x.to(args.device), batch)
        token_ids, att_mask, single_pred_label, single_arg_label, all_pred_label, \
        token_ids_p, att_mask_p, idx_phrs, entity_embs = batch
        pred_mask = bio.get_pred_mask(single_pred_label)
        entity_embs.requires_grad = True

        model.train()
        model.zero_grad()

        batch_loss, pred_loss, arg_loss = model(
            input_ids=token_ids,
            attention_mask=att_mask,
            input_ids_p=token_ids_p,
            attention_mask_p=att_mask_p,
            idx_phrs=idx_phrs,
            entity_embs=entity_embs,
            predicate_mask=pred_mask,
            total_pred_labels=all_pred_label,
            arg_labels=single_arg_label,
        )
        total_pred_loss += pred_loss.item()
        total_arg_loss += arg_loss.item()
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        trn_results = [total_pred_loss / (step + 1), total_arg_loss / (step + 1)]
        if step > epoch_steps:
            break

    utils.print_results(f"EPOCH{epoch} TRAIN",
                        trn_results, ["PRED LOSS", "ARG LOSS "])
    return trn_results, model
