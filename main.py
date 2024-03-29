# -*- coding:UTF-8 -*-
import os
import pynvml
import numpy as np


def get_best_gpu():
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    deviceMemory = []
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = int(np.argmax(deviceMemory))
    return best_device_index


best_gpu = str(get_best_gpu())
os.environ['CUDA_VISIBLE_DEVICES'] = best_gpu

import argparse
import torch
from utils import utils as utils
from utils.utils import SummaryManager, split_dataset, split_dataset1
from utils.emfh_args import get_emfh_args
from dataset import load_data
from train import train
from extract import extract
from test import do_eval
import datetime
import time
import sys
import traceback
from tqdm import tqdm
import logging

root_logger = logging.getLogger()
for h in root_logger.handlers:
    root_logger.removeHandler(h)

logging.basicConfig(format='%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')


def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--mode', default='cte')  # cte oie
    parser.add_argument('--save_path', default='./results')
    parser.add_argument('--bert_config', default='bert-base-cased', help='or bert-base-multilingual-cased')
    parser.add_argument('--trn_data_path', default='./datasets/openie4_train.pkl')
    parser.add_argument('--dev_data_path', nargs='+',
                        default=['./datasets/evaluate/oie2016_dev.pkl', './datasets/evaluate/carb_dev.pkl'])
    parser.add_argument('--dev_gold_path', nargs='+', default=['./evaluate_oie/OIE2016_dev.txt', './carb/CaRB_dev.tsv'])
    parser.add_argument('--log_path', default='./results')
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--visible_device', default="0")
    parser.add_argument('--summary_step', type=int, default=100)
    parser.add_argument('--use_lstm', nargs='?', const=True, default=False, type=utils.str2bool)
    parser.add_argument('--binary', nargs='?', const=True, default=False, type=utils.str2bool)
    parser.add_argument('--arg_n_labels', type=int, default=11)

    # hyper-parameters
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lstm_dropout', type=float, default=0.)
    parser.add_argument('--mh_dropout', type=float, default=0.2)
    parser.add_argument('--pred_clf_dropout', type=float, default=0.)
    parser.add_argument('--arg_clf_dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dev_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--n_arg_heads', type=int, default=8)
    parser.add_argument('--n_arg_layers', type=int, default=4)
    parser.add_argument('--pos_emb_dim', type=int, default=64)
    parser.add_argument('--trn_ent_emb_path', default='./datasets/knowledge/ent_emb_trn_multi.pkl')
    parser.add_argument('--eva_ent_emb_path', default='./datasets/knowledge/ent_emb_eva.pkl')
    parser.add_argument('--max_ent_num', type=int, default=2)

    # EMFH parameters
    parser.add_argument('--high_order', type=bool, default=False)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--mfb_k', type=int, default=10)
    parser.add_argument('--mfb_o', type=int, default=1024)
    parser.add_argument('--lstm_out_size', type=int, default=128)
    parser.add_argument('--frcn_feat_size', type=int, default=128)
    parser.add_argument('--dropout_r', type=float, default=0.1)
    parser.add_argument('--i_glimpses', type=int, default=2)
    parser.add_argument('--q_glimpses', type=int, default=2)
    parser.add_argument('--feat_size', type=int, default=1024)
    parser.add_argument('--n_layers', type=int, default=2)

    main_args = parser.parse_args()

    return main_args


def main(args):
    try:
        utils.set_seed(args.seed)
        logger.info('======Get OpenIE Models======')
        emfh_args = get_emfh_args(args)
        args.epochs = 100
        args.arg_n_labels = 9
        model = utils.get_models(
            bert_config=args.bert_config,
            pred_n_labels=args.pred_n_labels,
            arg_n_labels=args.arg_n_labels,
            n_arg_heads=args.n_arg_heads,
            n_arg_layers=args.n_arg_layers,
            lstm_dropout=args.lstm_dropout,
            mh_dropout=args.mh_dropout,
            pred_clf_dropout=args.pred_clf_dropout,
            arg_clf_dropout=args.arg_clf_dropout,
            pos_emb_dim=args.pos_emb_dim,
            use_lstm=args.use_lstm,
            device=args.device,
            emfh_args=emfh_args)
        logger.info('======Load OpenIE Training Set======')
        trn_loader = load_data(
            data_path=args.trn_data_path,
            batch_size=args.batch_size,
            max_len=args.max_len,
            tokenizer_config=args.bert_config,
            ent_emb_path=args.trn_ent_emb_path,
            max_ent_num=args.max_ent_num)
        args.total_steps = round(len(trn_loader) * args.epochs)
        args.warmup_steps = round(args.total_steps / 10)
        optimizer, scheduler = utils.get_train_modules(
            model=model,
            lr=args.learning_rate,
            total_steps=args.total_steps,
            warmup_steps=args.warmup_steps)
        model.zero_grad()
        model_oie = None
        logger.info("======OpenIE Training Starts======")
        for epoch in tqdm(range(1, args.epochs + 1), desc='epochs'):
            logger.info('\nEpoch-->{}'.format(str(epoch)))
            trn_results, model_oie = train(args, epoch, model, trn_loader, None, None, optimizer,
                                           scheduler)

        logger.info('======Get Constraint Models======')
        model = utils.get_cons_models(model_oie, device=args.device)
        eval_train, eval_valid, eval_test, eval_train_ent, eval_valid_ent, eval_test_ent, path_golds_valid, \
        path_golds_test = split_dataset(args.dev_data_path[0], args.eva_ent_emb_path)
        args.dev_gold_path = [path_golds_valid, path_golds_test]
        logger.info('======Load Constraint Training Set======')
        args.epochs = 100  # 100
        trn_loader = load_data(
            data_path=eval_train,
            batch_size=args.batch_size,
            max_len=args.max_len,
            tokenizer_config=args.bert_config,
            ent_emb_path=eval_train_ent,
            max_ent_num=args.max_ent_num,
            constraint=True)
        logger.info('======Load Constraint Validation Set======')
        args.dev_data_path = [eval_valid, eval_test]
        args.eva_ent_emb_path = [eval_valid_ent, eval_test_ent]
        dev_loaders = [
            load_data(
                data_path=cur_dev_path,
                batch_size=args.dev_batch_size,
                tokenizer_config=args.bert_config,
                train=False,
                ent_emb_path=cur_ent_emb,
                constraint=True)
            for cur_dev_path, cur_ent_emb in zip(args.dev_data_path, args.eva_ent_emb_path)]
        args.total_steps = round(len(trn_loader) * args.epochs)
        args.warmup_steps = round(args.total_steps / 10)
        optimizer, scheduler = utils.get_train_modules(
            model=model,
            lr=args.learning_rate,
            total_steps=args.total_steps,
            warmup_steps=args.warmup_steps)
        model.zero_grad()
        summarizer = SummaryManager(args)

        logger.info("======Constraint Training Starts======")
        best_result = None
        best_f1 = -1
        for epoch in tqdm(range(1, args.epochs + 1), desc='epochs'):
            logger.info('\nEpoch-->{}'.format(str(epoch)))
            trn_results, _ = train(args, epoch, model, trn_loader, dev_loaders, summarizer, optimizer,
                                   scheduler)

            dev_results = list()
            total_sum = 0
            for i_dev in range(len(dev_loaders)):
                dev_input = args.dev_data_path[i_dev]
                dev_gold = args.dev_gold_path[i_dev]
                dev_loader = dev_loaders[i_dev]
                dev_name = 'Valid' if i_dev == 0 else 'Test'
                output_path = os.path.join(args.save_path, f'epoch{epoch}_dev/end_epoch/{dev_name}')
                extract(args, model, dev_loader, output_path, True)
                dev_result = do_eval(output_path, dev_gold)
                if dev_result[0] > best_f1:
                    best_result = dev_result
                    best_f1 = dev_result[0]
                utils.print_results(f"EPOCH{epoch} Evaluate: {dev_name}",
                                    dev_result, ["F1  ", "PREC", "REC ", "AUC "])
                total_sum += dev_result[0] + dev_result[-1]
                dev_result.append(dev_result[0] + dev_result[-1])
                dev_results += dev_result
            model_name = utils.set_model_name(total_sum, epoch)
            if epoch % args.epochs == 0 and epoch != 0:
                torch.save(model, os.path.join(args.save_path, model_name))
        utils.print_results(f"Best Results",
                            best_result, ["F1  ", "PREC", "REC ", "AUC "])
        logger.info("Training Ended")
    except:
        logger.info(traceback.format_exc())
        return 0


if __name__ == '__main__':
    time_tag = time.strftime('%Y%m%d-%H%M%S')
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('root')

    start_time = datetime.datetime.now()

    main_args = parse_args()

    if main_args.mode == 'cte':
        print('Constrained Tuple Extraction')
    elif main_args.mode == 'oie':
        print('Open Information Extraction')

    main_args.save_path = './results/log-%s' % time_tag
    main_args.log_path = main_args.save_path
    main_args.visible_device = best_gpu

    if not os.path.exists(main_args.log_path):
        os.makedirs(main_args.log_path)

    main_args = utils.clean_config(main_args)

    path_log = os.path.join(main_args.log_path, "train-{}.log".format(time_tag))
    logger.addHandler(logging.FileHandler(path_log, 'w'))

    logger.info('Current PID: ' + str(os.getpid()))
    logger.info('Run in GPU: ' + main_args.visible_device)

    main(main_args)

    end_time = datetime.datetime.now()
    logger.info('Total time taken: {} s'.format((end_time - start_time).seconds))
