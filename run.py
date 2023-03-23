#!/usr/bin/env python
# coding:utf-8

import math
import os
import sys
import time
import warnings
from copy import deepcopy

import numpy as np
import torch
from transformers import AutoTokenizer

import utils.logger as logger
from model import classifier
from utils.configuration import Configuration
from utils.dataloader import HTCDataset, collate_fn
from utils.file_and_ckpt import *
from utils.loss import *
from utils.metric import *


def train(config):
    # READ FILES ACCORDING TO CONFIG
    label_ids = make_label_indices(config)
    hierarchy = read_hierarchy(config, label_ids)
    label_sequences = make_label_sequences(hierarchy, label_ids)
    
    # DATASET AND DATALOADER GENERATION
    tokenizer = AutoTokenizer.from_pretrained(config.model.embedding.type)
    collate_function = collate_fn(config, tokenizer, label_ids)
    train_dataset = HTCDataset(config, 'train', label_ids)
    val_dataset = HTCDataset(config, 'val', label_ids)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, config.training.batch_size, shuffle=True, num_workers=config.device.num_workers,
                                                   collate_fn=collate_function, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, config.training.batch_size, shuffle=False, num_workers=config.device.num_workers,
                                                 collate_fn=collate_function, drop_last=False)
    logger.info('DATASET LOADED')
    
    # MODEL, CRITERION, OPTIMIZER AND SCHEDULER DEFINITION
    model = classifier.TextClassifier(config, label_ids)
    criterion = FocalLoss()
    penalty = LabelContradictionPenalty(config, hierarchy)
    optimizer = getattr(torch.optim, config.training.optimizer.type)(params=model.parameters(), lr=config.training.optimizer.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.training.schedule.decay)
    logger.info('MODEL GENERATED')
    logger.info('NUMBER OF PARAMETERS: %d' % sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()))

    # LOAD CHECKPOINT FROM DIRECTORY
    best_epoch = -1
    best_performances = {'micro-f1': 0., 'macro-f1': 0.}
    epoch = 0
    checkpoint = config.path.initial_checkpoint
    if os.path.isfile(checkpoint):
        logger.info('CHECKPOINT LOADING FROM %s' % checkpoint)
        epoch, best_performances = load_checkpoint(checkpoint, model, optimizer)
        logger.info('CHECKPOINT LOADED FROM EPOCH %d' % (epoch - 1))
        logger.info('\tMICRO-F1: %.5f | MACRO-F1: %.5f' % (best_performances['micro-f1'], best_performances['macro-f1']))
        best_epoch = epoch - 1
    
    # USE PARALLEL GPUS
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
        logger.info('CUDA AVAILABLE: DATA PARALLEL MODEL DEFINED')
        optim_state_dict = optimizer.state_dict()
        for state in optim_state_dict['state'].values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    else:
        logger.info('CUDA UNAVAILABLE: RUNNING ON CPU')
    
    # START TRAINING
    suboptimal_epochs = 0
    isnan = False
    for epoch in range(epoch, config.training.num_epochs):
        if hasattr(config.training, "finetune") and config.training.finetune.tune and epoch >= config.training.finetune.after:
            for p in model.module.text_embedder.parameters():
                p.requires_grad = True
            model = model.module
            model.is_finetune = -1 * model.is_finetune
            optimizer = getattr(torch.optim, config.training.optimizer.type)(params=model.parameters(), lr=config.training.finetune.learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.training.schedule.decay)
            model = torch.nn.parallel.DataParallel(model.cuda())
        logger.info('EPOCH %d START' % epoch)
        start_time = time.time()
        model.train()
        total_loss = 0.
        epoch_labels = []
        epoch_preds = []
        if hasattr(config.model, 'capsule_setting'):
            model.module.information_aggregation.classifier.scaler = 1.
        for step, batch in enumerate(train_dataloader):
            step_start = time.time()
            # FORWARD
            preds, _ = model(batch)
            loss = criterion(preds, batch[3].to(preds.device))
            if penalty.penalty_weight > 0:
                loss += penalty(preds)
            total_loss += loss.item()
            epoch_labels.append(batch[3])
            epoch_preds.append(preds.detach().cpu())
            # BACKWARD
            for p in model.parameters():
                p.grad = None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            logger.info('EPOCH %d STEP %d / %d LOSS %f (%.3fs)' %
                        (epoch, step + 1, len(train_dataset) // config.training.batch_size, loss.item(), time.time() - step_start))
            isnan = math.isnan(loss.item())
            if isnan:
                logger.info('NAN LOSS RETURNED, TRAINING STOPPED')
                break
        # F1 SCORE CALCULATION
        epoch_labels = torch.cat(epoch_labels, dim=0).to(torch.bool).cpu().numpy()
        epoch_preds = torch.cat(epoch_preds, dim=0).cpu().numpy()
        epoch_predicted_labels = epoch_preds >= 0.5
        train_performances = f1_scores(epoch_labels, epoch_preds, epoch_predicted_labels, label_ids, label_sequences, [0, 0, 0])
        for k in train_performances.keys():
            if np.isnan(train_performances[k]):
                train_performances[k] = 0.
        logger.info('EPOCH %d TRAINING FINISHED. LOSS: %f | MICRO-F1: %f | MACRO-F1: %f' %
                    (epoch, total_loss / (len(train_dataset) // config.training.batch_size), train_performances['micro-f1'], train_performances['macro-f1']))
        
        # VALIDATION
        total_loss = 0.
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                if step == 0 and hasattr(config.model, 'capsule_setting'):
                    _, tr_p = model(batch)
                    model.eval()
                    _, va_p = model(batch)
                    model.module.information_aggregation.classifier.scaler = (torch.norm(tr_p) / torch.norm(va_p)).item()
                preds, _ = model(batch)
                loss = criterion(preds, batch[3].to(preds.device))
                if penalty.penalty_weight > 0:
                    loss += penalty(preds)
                total_loss += loss.item()
                val_preds.append(preds.detach().cpu())
                val_labels.append(batch[3])
            # F1 SCORE CALCULATION
            val_labels = torch.cat(val_labels, dim=0).to(torch.bool).cpu().numpy()
            val_preds = torch.cat(val_preds, dim=0).cpu().numpy()
            val_predicted_labels = val_preds >= 0.5
            val_performances = f1_scores(val_labels, val_preds, val_predicted_labels, label_ids, label_sequences, [0, 0, 0])
            logger.info('EPOCH %d VALIDATION FINISHED. LOSS: %f | MICRO-F1: %f | MACRO-F1: %f' %
                        (epoch, total_loss / (len(val_dataset) // config.training.batch_size), val_performances['micro-f1'], val_performances['macro-f1']))
        # PERFORMANCE COMPARISON WITH PREVIOUS CHECKPOINTS
        for k in val_performances.keys():
            if np.isnan(val_performances[k]):
                val_performances[k] = 0.
        if isnan:
            logger.info('NAN LOSS WHILE TRAINING: EARLY STOPPING')
            break
        if val_performances['micro-f1'] < best_performances['micro-f1']:
            suboptimal_epochs += 1
            # LEARNING RATE DECAY
            if suboptimal_epochs % config.training.schedule.patience == 0:
                scheduler.step()
                logger.info('%d CONSECUTIVE EPOCHS WITHOUT PERFORMANCE IMPROVEMENT: LEARNING RATE DECAYED TO %f' % (suboptimal_epochs, optimizer.param_groups[0]['lr']))
            # EARLY STOPPING
            if suboptimal_epochs == config.training.schedule.early_stopping:
                logger.info('%d CONSECUTIVE EPOCHS WITHOUT PERFORMANCE IMPROVEMENT: EARLY STOPPING' % suboptimal_epochs)
                break
        else:
            # BEST CHECKPOINT UPDATE
            suboptimal_epochs = 0
            logger.info('MICRO-F1 IMPROVED FROM %.5f (EPOCH %d) TO %.5f (EPOCH %d)' 
                        % (best_performances['micro-f1'], best_epoch, val_performances['micro-f1'], epoch))
            best_performances['micro-f1'] = val_performances['micro-f1']
            best_epoch = epoch
            save_checkpoint(os.path.join(config.path.checkpoints, 'best_micro_f1.ckpt'), epoch, val_performances, model, optimizer)
        logger.info('EPOCH %d COMPLETED (%d SECONDS)' % (epoch, time.time() - start_time))
    return


def hyperparameter_search(config):
    # TEST DATASET AND DATALOADER GENERATED
    label_ids = make_label_indices(config)
    hierarchy = read_hierarchy(config, label_ids)
    label_sequences = make_label_sequences(hierarchy, label_ids)
    tokenizer = AutoTokenizer.from_pretrained(config.model.embedding.type)
    collate_function = collate_fn(config, tokenizer, label_ids)
    val_dataset = HTCDataset(config, 'val', label_ids)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, config.training.batch_size, shuffle=False, num_workers=config.device.num_workers,
                                                 collate_fn=collate_function, drop_last=False)

    # MODEL DEFINITION
    model = classifier.TextClassifier(config, label_ids)
    best_model = os.path.join(config.path.checkpoints, 'best_micro_f1.ckpt')
    best_epoch = load_checkpoint(best_model, model, None, 'test')

    # USE PARALLEL GPUS
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
        logger.info('CUDA AVAILABLE: DATA PARALLEL MODEL DEFINED')
    else:
        logger.info('CUDA UNAVAILABLE: RUNNING ON CPU')
    logger.info('BEST MICRO F1 CHECKPOINT FROM EPOCH %d LOADED FOR HYPERPARAMETER SEARCH' % best_epoch)

    best_setting = dict()
    model.eval()
    with torch.no_grad():
        val_labels = []
        val_preds = []
        for step, batch in enumerate(val_dataloader):
            if step == 0 and hasattr(config.model, 'capsule_setting'):
                model.module.information_aggregation.classifier.scaler = 1.
                model.train()
                _, tr_p = model(batch)
                model.eval()
                _, va_p = model(batch)
                model.module.information_aggregation.classifier.scaler = (torch.norm(tr_p) / torch.norm(va_p)).item()
            preds, _ = model(batch)
            val_preds.append(preds.detach().cpu())
            val_labels.append(batch[3])
        val_labels = torch.cat(val_labels, dim=0).to(torch.bool).cpu().numpy()
        val_preds = torch.cat(val_preds, dim=0).cpu().numpy()

    logger.info('BEGIN HYPERPARAMETER SEARCH ON CONFIDENCE THRESHOLD')
    best_perf = 0.
    for thr in np.arange(0.3, 0.85, 0.05):
        val_predicted_labels = val_preds >= thr
        val_performance = f1_scores(val_labels, val_preds, val_predicted_labels, label_ids, label_sequences, [0, 0, 0])['micro-f1']
        logger.info('CONFIDENCE THRESHOLD = %f: PERFORMANCE (MICRO-F1) = %f' % (thr, val_performance))
        if best_perf < val_performance:
            best_perf = val_performance
            best_setting['confidence_threhold'] = thr
    logger.info('BEST SETTING: CONFIDENCE THRESHOLD = %f WITH MICRO-F1 = %f' % (best_setting['confidence_threhold'], best_perf))

    logger.info('BEGIN HYPERPARAMETER SEARCH ON POSTPROCESSING METHODS')
    best_perf = 0.
    val_predicted_labels = val_preds >= best_setting['confidence_threhold']
    for s1 in ['none', 'remove', 'connect']:
        for s2 in ['none', 'remove']:
            for s3 in ['none', 'argmax_leaf', 'argmax_path']:
                postproc = [{'none': 0, 'remove': 1, 'connect': 2}[s1],
                            {'none': 0, 'remove': 1}[s2],
                            {'none': 0, 'argmax_leaf': 1, 'argmax_path': 2}[s3]]
                val_performance = f1_scores(val_labels, val_preds, val_predicted_labels, label_ids, label_sequences, postproc)['micro-f1']
                logger.info('POSTPROCESSING %s: PERFORMANCE (MICRO-F1) = %f' % (postproc, val_performance))
                if best_perf < val_performance:
                    best_perf = val_performance
                    best_setting['postprocessing'] = postproc
    logger.info('BEST SETTING: POSTPROCESSING = %s WITH MICRO-F1 = %f' % (best_setting['postprocessing'], best_perf))
    
    return best_setting


def test(config, best_setting=None):
    if best_setting is None:
        best_setting = {'confidence_threhold': 0.5, 'postprocessing': [0, 0, 0]}
    # TEST DATASET AND DATALOADER GENERATED
    label_ids = make_label_indices(config)
    hierarchy = read_hierarchy(config, label_ids)
    label_sequences = make_label_sequences(hierarchy, label_ids)
    tokenizer = AutoTokenizer.from_pretrained(config.model.embedding.type)
    collate_function = collate_fn(config, tokenizer, label_ids)
    test_dataset = HTCDataset(config, 'test', label_ids)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, config.training.batch_size, shuffle=False, num_workers=config.device.num_workers,
                                                  collate_fn=collate_function, drop_last=False)

    # MODEL DEFINITION
    model = classifier.TextClassifier(config, label_ids)
    best_model = os.path.join(config.path.checkpoints, 'best_micro_f1.ckpt')
    best_epoch = load_checkpoint(best_model, model, None, 'test')

    # USE PARALLEL GPUS
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
        logger.info('CUDA AVAILABLE: DATA PARALLEL MODEL DEFINED')
    else:
        logger.info('CUDA UNAVAILABLE: RUNNING ON CPU')
    logger.info('BEST MICRO F1 CHECKPOINT FROM EPOCH %d LOADED FOR TEST' % best_epoch)

    test_labels = []
    test_preds = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            if step == 0 and hasattr(config.model, 'capsule_setting'):
                model.module.information_aggregation.classifier.scaler = 1.
                model.train()
                _, tr_p = model(batch)
                model.eval()
                _, va_p = model(batch)
                model.module.information_aggregation.classifier.scaler = (torch.norm(tr_p) / torch.norm(va_p)).item()
            preds, _ = model(batch)
            test_preds.append(preds.detach().cpu())
            test_labels.append(batch[3])
        test_labels = torch.cat(test_labels, dim=0).to(torch.bool).cpu().numpy()
        logger.info('best path %s' % select_best_path(test_labels, torch.cat(test_preds, dim=0).cpu(), label_ids, label_sequences))
        logger.info('best leaf %s' % select_best_leaf(test_labels, torch.cat(test_preds, dim=0).cpu(), label_ids, label_sequences))
        logger.info('autoreg %s' % select_best_path_autoregressive(test_labels, torch.cat(test_preds, dim=0).cpu(), label_ids, label_sequences))
        test_preds = torch.cat(test_preds, dim=0).cpu().numpy()
        test_predicted_labels = test_preds >= best_setting['confidence_threhold']
        test_performances = f1_scores(test_labels, test_preds, test_predicted_labels, label_ids, label_sequences, best_setting['postprocessing'])
        logger.info('TEST FOR BEST MICRO F1 CHECKPOINT (EPOCH %d) FINISHED. MICRO-F1: %f | MACRO-F1: %f'
                    % (best_epoch, test_performances['micro-f1'], test_performances['macro-f1']))

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config_path = sys.argv[1]
    assert os.path.isfile(config_path), "CONFIGURATION FILE DOES NOT EXIST"
    with open(config_path, 'r') as fin:
        config = json.load(fin)
    config = Configuration(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in config.device.cuda])
    os.environ['TOKENIZERS_PARALLELISM'] = 'true' if config.device.num_workers > 0 else 'false'
    torch.manual_seed(1227)
    torch.cuda.manual_seed(1227)
    logger.add_filehandler(config.path.log)
    logger.logging_verbosity(1)
    torch.multiprocessing.set_sharing_strategy('file_system')

    if not os.path.isdir(config.path.checkpoints):
        os.mkdir(config.path.checkpoints)

    train(config)
    best_setting = hyperparameter_search(config)
    logger.info('TESTING CLASSIFIER')
    test(config, best_setting=best_setting)
