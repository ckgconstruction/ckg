import argparse
import torch
import os
import sys
import random
import numpy as np
import pickle
import pandas as pd
import json
import copy

from model import InteractionAware

from transformers import get_linear_schedule_with_warmup, AdamW
import logging
from tqdm import tqdm
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
import torch.nn as nn

logger = logging.getLogger('root')


def r_list(data):
    r_data = []
    len_row = len(data)
    len_col = len(data[0])
    for i in range(len_col):
        col = []
        for j in range(len_row):
            col.append(data[j][i])
        r_data.append(col)
    return r_data


def split_dataset(dev_data_path, eva_ent_emb_path, ratio_train=0.6, ratio_valid=0.2, ratio_test=0.2):
    eval_data = load_pkl(dev_data_path)
    eval_ent_emb = load_pkl(eva_ent_emb_path)
    assert len(eval_data['sentences']) == len(eval_ent_emb['entity_embs'])

    del eval_data['max_over_case']
    del eval_data['rel_pos_malformed']
    train = None
    valid = None
    test = None
    ss = ShuffleSplit(n_splits=1, test_size=(ratio_valid + ratio_test), random_state=0)
    for train_index, temp_index in ss.split(eval_data['sentences']):
        train_index = np.concatenate((train_index, np.asarray(temp_index[0:1])))
        temp_index = temp_index[1:]
        train = {
            'sentences': list(),
            'sentences_phr': list(),
            'tokens': list(),
            'phrase_tokens': list(),
            'single_pred_labels': list(),
            'single_arg_labels': list(),
            'all_pred_labels': list(),
            'phrase_idx': list(),
            'ners': list()
        }
        train_ent = {
            'entities': list(),
            'entity_embs': list()
        }
        for train_id in train_index:
            for key in eval_data.keys():
                train[key].append(eval_data[key][int(train_id)])
            for key in eval_ent_emb.keys():
                train_ent[key].append(eval_ent_emb[key][int(train_id)])
        temp = {
            'sentences': list(),
            'sentences_phr': list(),
            'tokens': list(),
            'phrase_tokens': list(),
            'single_pred_labels': list(),
            'single_arg_labels': list(),
            'all_pred_labels': list(),
            'phrase_idx': list(),
            'ners': list()
        }
        temp_ent = {
            'entities': list(),
            'entity_embs': list()
        }
        for temp_id in temp_index:
            for key in eval_data.keys():
                temp[key].append(eval_data[key][int(temp_id)])
            for key in eval_ent_emb.keys():
                temp_ent[key].append(eval_ent_emb[key][int(temp_id)])
        ss2 = ShuffleSplit(n_splits=1, test_size=(ratio_test / (ratio_valid + ratio_test)),
                           random_state=0)
        for valid_index, test_index in ss2.split(temp['sentences']):

            valid = {
                'sentences': list(),
                'sentences_phr': list(),
                'tokens': list(),
                'phrase_tokens': list(),
                'single_pred_labels': list(),
                'single_arg_labels': list(),
                'all_pred_labels': list(),
                'phrase_idx': list(),
                'ners': list()
            }
            valid_ent = {
                'entities': list(),
                'entity_embs': list()
            }
            for valid_id in valid_index:
                for key in temp.keys():
                    valid[key].append(temp[key][int(valid_id)])
                for key in temp_ent.keys():
                    valid_ent[key].append(temp_ent[key][int(valid_id)])
            # 测试集
            test = {
                'sentences': list(),
                'sentences_phr': list(),
                'tokens': list(),
                'phrase_tokens': list(),
                'single_pred_labels': list(),
                'single_arg_labels': list(),
                'all_pred_labels': list(),
                'phrase_idx': list(),
                'ners': list()
            }
            test_ent = {
                'entities': list(),
                'entity_embs': list()
            }
            for test_id in test_index:
                for key in temp.keys():
                    test[key].append(temp[key][int(test_id)])
                for key in temp_ent.keys():
                    test_ent[key].append(temp_ent[key][int(test_id)])
    assert len(train['sentences']) == len(train_ent['entities'])
    assert len(valid['sentences']) == len(valid_ent['entities'])
    assert len(test['sentences']) == len(test_ent['entities'])
    train_new = {
        'sentences': list(),
        'sentences_phr': list(),
        'tokens': list(),
        'phrase_tokens': list(),
        'single_pred_labels': list(),
        'single_arg_labels': list(),
        'all_pred_labels': list(),
        'phrase_idx': list(),
        'ners': list()
    }
    train_ent_new = {
        'entities': list(),
        'entity_embs': list()
    }
    for i in range(len(train['sentences'])):
        for j in range(len(train['single_pred_labels'][i])):
            for key in train.keys():
                if key in ['sentences', 'sentences_phr', 'tokens', 'phrase_tokens', 'phrase_idx', 'ners']:
                    train_new[key].append(train[key][i])
                elif key in ['single_pred_labels', 'single_arg_labels', 'all_pred_labels']:
                    train_new[key].append(train[key][i][j])
            for key in train_ent.keys():
                train_ent_new[key].append(train_ent[key][i])
    assert len(train_new['sentences']) == len(train_new['single_pred_labels']) and \
           len(train_new['sentences']) == len(train_new['single_arg_labels']) and \
           len(train_new['sentences']) == len(train_new['all_pred_labels']) and \
           len(train_new['sentences']) == len(train_ent_new['entities'])
    # 生成评测集的金标注
    path_golds_valid = generate_golds(valid, '../datasets/gold/gold_valid.txt')
    path_golds_test = generate_golds(test, '../datasets/gold/gold_test.txt')
    return train_new, valid, test, train_ent_new, valid_ent, test_ent, path_golds_valid, path_golds_test  # 1305, 434, 435


def generate_golds(data, path_out):
    with open('../datasets/gold/gold.txt', 'r', encoding='utf-8') as file:
        golds_all = file.readlines()
    sentences = data['sentences']
    golds = []
    for i_sent, sent in enumerate(tqdm(sentences)):
        t = 0
        for i_gold, gold in enumerate(golds_all):
            if sent[:30] == gold[:30]:
                golds.append(gold)
                t = 1
    with open(path_out, 'w', encoding='utf-8') as file_out:
        for gold in golds:
            file_out.write(gold)
    print()
    return path_out


def split_dataset1(dev_data_path, eva_ent_emb_path, ratio_train=0.6, ratio_valid=0.2, ratio_test=0.2):
    eval_data = load_pkl(dev_data_path)
    eval_ent_emb = load_pkl(eva_ent_emb_path)
    assert len(eval_data['sentences']) == len(eval_ent_emb['entity_embs'])

    del eval_data['max_over_case']
    del eval_data['rel_pos_malformed']
    train = None
    valid = None
    test = None
    ss = ShuffleSplit(n_splits=1, test_size=(ratio_valid + ratio_test), random_state=0)
    for train_index, temp_index in ss.split(eval_data['sentences']):
        train_index = np.concatenate((train_index, np.asarray(temp_index[0:1])))  # 1304
        temp_index = temp_index[1:]
        train = {
            'sentences': list(),
            'sentences_phr': list(),
            'tokens': list(),
            'phrase_tokens': list(),
            'single_pred_labels': list(),
            'single_arg_labels': list(),
            'all_pred_labels': list(),
            'phrase_idx': list(),
            'ners': list()
        }
        train_ent = {
            'entities': list(),
            'entity_embs': list()
        }
        for train_id in train_index:
            for key in eval_data.keys():
                train[key].append(eval_data[key][int(train_id)])
            for key in eval_ent_emb.keys():
                train_ent[key].append(eval_ent_emb[key][int(train_id)])
        temp = {
            'sentences': list(),
            'sentences_phr': list(),
            'tokens': list(),
            'phrase_tokens': list(),
            'single_pred_labels': list(),
            'single_arg_labels': list(),
            'all_pred_labels': list(),
            'phrase_idx': list(),
            'ners': list()
        }
        temp_ent = {
            'entities': list(),
            'entity_embs': list()
        }
        for temp_id in temp_index:
            for key in eval_data.keys():
                temp[key].append(eval_data[key][int(temp_id)])
            for key in eval_ent_emb.keys():
                temp_ent[key].append(eval_ent_emb[key][int(temp_id)])
        ss2 = ShuffleSplit(n_splits=1, test_size=(ratio_test / (ratio_valid + ratio_test)),
                           random_state=0)
        for valid_index, test_index in ss2.split(temp['sentences']):
            valid = {
                'sentences': list(),
                'sentences_phr': list(),
                'tokens': list(),
                'phrase_tokens': list(),
                'single_pred_labels': list(),
                'single_arg_labels': list(),
                'all_pred_labels': list(),
                'phrase_idx': list(),
                'ners': list()
            }
            valid_ent = {
                'entities': list(),
                'entity_embs': list()
            }
            for valid_id in valid_index:
                for key in train.keys():
                    valid[key].append(train[key][int(valid_id)])
                for key in train_ent.keys():
                    valid_ent[key].append(train_ent[key][int(valid_id)])
            test = {
                'sentences': list(),
                'sentences_phr': list(),
                'tokens': list(),
                'phrase_tokens': list(),
                'single_pred_labels': list(),
                'single_arg_labels': list(),
                'all_pred_labels': list(),
                'phrase_idx': list(),
                'ners': list()
            }
            test_ent = {
                'entities': list(),
                'entity_embs': list()
            }
            for test_id in test_index:
                for key in train.keys():
                    test[key].append(train[key][int(test_id)])
                for key in train_ent.keys():
                    test_ent[key].append(train_ent[key][int(test_id)])
    assert len(train['sentences']) == len(train_ent['entities'])
    assert len(valid['sentences']) == len(valid_ent['entities'])
    assert len(test['sentences']) == len(test_ent['entities'])
    # 将训练集修改为以元组为单位
    train_new = {
        'sentences': list(),
        'sentences_phr': list(),
        'tokens': list(),
        'phrase_tokens': list(),
        'single_pred_labels': list(),
        'single_arg_labels': list(),
        'all_pred_labels': list(),
        'phrase_idx': list(),
        'ners': list()
    }
    train_ent_new = {
        'entities': list(),
        'entity_embs': list()
    }
    for i in range(len(train['sentences'])):
        for j in range(len(train['single_pred_labels'][i])):
            for key in train.keys():
                if key in ['sentences', 'sentences_phr', 'tokens', 'phrase_tokens', 'phrase_idx', 'ners']:
                    train_new[key].append(train[key][i])
                elif key in ['single_pred_labels', 'single_arg_labels', 'all_pred_labels']:
                    train_new[key].append(train[key][i][j])
            for key in train_ent.keys():
                train_ent_new[key].append(train_ent[key][i])
    assert len(train_new['sentences']) == len(train_new['single_pred_labels']) and \
           len(train_new['sentences']) == len(train_new['single_arg_labels']) and \
           len(train_new['sentences']) == len(train_new['all_pred_labels']) and \
           len(train_new['sentences']) == len(train_ent_new['entities'])
    return train_new, valid, test, train_ent_new, valid_ent, test_ent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def clean_config(config):
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    config.device = device
    config.pred_n_labels = 3
    config.arg_n_labels = 9
    os.makedirs(config.save_path, exist_ok=True)
    return config


def get_cons_models(model, device):
    model.arg_n_labels = 13
    model.arg_classifier = nn.Linear(1776, 13)
    return model.to(device)


def get_models(bert_config,
               pred_n_labels=3,
               arg_n_labels=9,
               n_arg_heads=8,
               n_arg_layers=4,
               lstm_dropout=0.3,
               mh_dropout=0.1,
               pred_clf_dropout=0.,
               arg_clf_dropout=0.3,
               pos_emb_dim=64,
               use_lstm=False,
               device=None,
               emfh_args=None):
    if not use_lstm:
        return InteractionAware(
            bert_config=bert_config,
            mh_dropout=mh_dropout,
            pred_clf_dropout=pred_clf_dropout,
            arg_clf_dropout=arg_clf_dropout,
            n_arg_heads=n_arg_heads,
            n_arg_layers=n_arg_layers,
            pos_emb_dim=pos_emb_dim,
            pred_n_labels=pred_n_labels,
            arg_n_labels=arg_n_labels,
            emfh_args=emfh_args).to(device)


def save_pkl(path, file):
    with open(path, 'wb') as f:
        pickle.dump(file, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_word2piece(sentence, tokenizer):
    words = sentence.split(' ')
    word2piece = {idx: list() for idx in range(len(words))}
    sentence_pieces = list()
    piece_idx = 1
    for word_idx, word in enumerate(words):
        pieces = tokenizer.tokenize(word)
        sentence_pieces += pieces
        for piece_idx_added, piece in enumerate(pieces):
            word2piece[word_idx].append(piece_idx + piece_idx_added)
        piece_idx += len(pieces)
    assert len(sentence_pieces) == piece_idx - 1
    return word2piece


def get_train_modules(model,
                      lr,
                      total_steps,
                      warmup_steps):
    optimizer = AdamW(
        model.parameters(), lr=lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)
    return optimizer, scheduler


class SummaryManager:
    def __init__(self, config):
        self.config = config
        columns = ['epoch', 'train_predicate_loss', 'train_argument_loss']
        for i in range(2):
            cur_dev_name = 'ConstraintExtraction' + str(i)
            for metric in ['f1', 'prec', 'rec', 'auc', 'sum']:
                columns.append(f'{cur_dev_name}_{metric}')
        columns.append('total_sum')
        self.result_df = pd.DataFrame(columns=columns)
        self.save_df()

    def save_config(self, display=False):
        if display:
            for key, value in self.config.__dict__.items():
                logger.info("{}: {}".format(key, value))
        copied = copy.deepcopy(self.config)
        copied.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(os.path.join(copied.save_path, 'config.json'), 'w') as fp:
            json.dump(copied.__dict__, fp, indent=4)

    def save_results(self, results):
        self.result_df = pd.read_csv(os.path.join(self.config.save_path, 'train_results.csv'))
        self.result_df.loc[len(self.result_df.index)] = results
        self.save_df()

    def save_df(self):
        self.result_df.to_csv(os.path.join(self.config.save_path, 'train_results.csv'), index=False)


def set_model_name(dev_results, epoch, step=None):
    if step is not None:
        return "model-epoch{}-step{}-score{:.4f}.bin".format(epoch, step, dev_results)
    else:
        return "model-epoch{}-end-score{:.4f}.bin".format(epoch, dev_results)


def print_results(message, results, names):
    try:
        logger.info(f"===== {message} =====")
        for result, name in zip(results, names):
            logger.info("{}: {:.5f}".format(name, result))
    except:
        return
