# /usr/bin/env python
# coding=utf-8
"""train with valid"""
from model import BertForRE
from dataloader import CustomDataLoader
from evaluate import evaluate
from optimization import BertAdam
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import argparse
from tqdm import trange
import logging
import random
from transformers import BertConfig
import torch
import os
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


# load args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020,
                    help="random seed for initialization")
parser.add_argument('--ex_index', type=str, default=1)
parser.add_argument('--corpus_type', type=str, default="Job",
                    help="NYT, WebNLG, NYT*, WebNLG*")
parser.add_argument('--device_id', type=int, default=0, help="GPU index")
parser.add_argument('--epoch_num', required=True,
                    type=int, help="number of epochs")
parser.add_argument('--multi_gpu', action='store_true',
                    help="ensure multi-gpu training")
parser.add_argument('--restore_file', default=None,
                    help="name of the file containing weights to reload")

parser.add_argument('--corres_threshold', type=float,
                    default=0.5, help="threshold of global correspondence")
parser.add_argument('--rel_threshold', type=float, default=0.5,
                    help="threshold of relation judgement")
parser.add_argument('--ensure_corres', action='store_true',
                    help="correspondence ablation")
parser.add_argument('--ensure_rel', action='store_true',
                    help="relation judgement ablation")
parser.add_argument('--emb_fusion', type=str,
                    default="concat", help="way to embedding")

parser.add_argument('--num_negs', type=int, default=4,
                    help="number of negative sample when ablate relation judgement")


def train(model, data_iterator, optimizer, params, ex_params):
    """Train the model one epoch
    """
    # set model to training mode
    model.train()

    loss_avg = utils.RunningAverage()
    loss_avg_seq = utils.RunningAverage()
    loss_avg_mat = utils.RunningAverage()
    loss_avg_rel = utils.RunningAverage()
   

    # Use tqdm for progress bar
    # one epoch
    t = trange(len(data_iterator), ascii=True)
    
    for step, _ in enumerate(t):
        # fetch the next training batch
        batch = next(iter(data_iterator))
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, attention_mask, seq_tags, relations, corres_tags, rel_tags = batch

        # compute model output and loss
        loss, loss_seq, loss_mat, loss_rel = model(input_ids, attention_mask=attention_mask, seq_tags=seq_tags,
                                                   potential_rels=relations, corres_tags=corres_tags, rel_tags=rel_tags,
                                                   ex_params=ex_params)

        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if params.gradient_accumulation_steps > 1:
            loss = loss / params.gradient_accumulation_steps

        # back-prop
        loss.backward()

        if (step + 1) % params.gradient_accumulation_steps == 0:
            # performs updates using calculated gradients
            optimizer.step()
            model.zero_grad()

        # update the average loss
        loss_avg.update(loss.item() * params.gradient_accumulation_steps)
        loss_avg_seq.update(loss_seq.item())
        loss_avg_mat.update(loss_mat.item())
        loss_avg_rel.update(loss_rel.item())
        # 右边第一个0为填充数，第二个5为数字个数为5位，第三个3为小数点有效数为3，最后一个f为数据类型为float类型。
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()),
                      loss_seq='{:05.3f}'.format(loss_avg_seq()),
                      loss_mat='{:05.3f}'.format(loss_avg_mat()),
                      loss_rel='{:05.3f}'.format(loss_avg_rel()))

    return loss_avg(), loss_avg_seq(), loss_avg_mat(), loss_avg_rel()


def val(model, data_iterator, optimizer, params, ex_params):
    """Train the model one epoch
    """
    # set model to training mode
    model.eval()

    val_loss_avg = utils.RunningAverage()
    val_loss_avg_seq = utils.RunningAverage()
    val_loss_avg_mat = utils.RunningAverage()
    val_loss_avg_rel = utils.RunningAverage()
   

    # Use tqdm for progress bar
    # one epoch
    val_t = trange(len(data_iterator), ascii=True)
    
    for step, _ in enumerate(val_t):
        # fetch the next training batch
        batch = next(iter(data_iterator))
        batch = tuple(val_t.to(params.device) for val_t in batch)
        input_ids, attention_mask, seq_tags, relations, corres_tags, rel_tags = batch
        
        # compute model output and loss
        loss, loss_seq, loss_mat, loss_rel = model(input_ids, attention_mask=attention_mask, seq_tags=seq_tags,
                                                  potential_rels=relations, corres_tags=corres_tags, rel_tags=rel_tags,
                                                  ex_params=ex_params)

        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if params.gradient_accumulation_steps > 1:
            loss = loss / params.gradient_accumulation_steps


        # update the average loss
        val_loss_avg.update(loss.item() * params.gradient_accumulation_steps)
        val_loss_avg_seq.update(loss_seq.item())
        val_loss_avg_mat.update(loss_mat.item())
        val_loss_avg_rel.update(loss_rel.item())
        # 右边第一个0为填充数，第二个5为数字个数为5位，第三个3为小数点有效数为3，最后一个f为数据类型为float类型。
        val_t.set_postfix(val_loss='{:05.3f}'.format(val_loss_avg()),
                      val_loss_seq='{:05.3f}'.format(val_loss_avg_seq()),
                      val_loss_mat='{:05.3f}'.format(val_loss_avg_mat()),
                      val_loss_rel='{:05.3f}'.format(val_loss_avg_rel()))
                      
    return val_loss_avg(), val_loss_avg_seq(), val_loss_avg_mat(), val_loss_avg_rel()


def train_and_evaluate(model, params, ex_params, restore_file=None):
    """Train the model and evaluate every epoch."""
    # Load training data and val data
    dataloader = CustomDataLoader(params)
    train_loader = dataloader.get_dataloader(
        data_sign='train', ex_params=ex_params)
    val_loader = dataloader.get_dataloader(
        data_sign='val', ex_params=ex_params)
    temp_loader = dataloader.get_dataloader(
        data_sign='temp', ex_params=ex_params)

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            params.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        # 读取checkpoint
        model, optimizer = utils.load_checkpoint(restore_path)
    
    model.to(params.device)
    # parallel model
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    # fine-tuning
    param_optimizer = list(model.named_parameters())
    # pretrain model param
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]
    # downstream model param
    param_downstream = [(n, p) for n, p in param_optimizer if 'bert' not in n]
    no_decay = ['bias', 'LayerNorm', 'layer_norm']
    optimizer_grouped_parameters = [
        # pretrain model param
        {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate, 'lr': params.fin_tuning_lr
         },
        {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': params.fin_tuning_lr
         },
        # downstream model
        {'params': [p for n, p in param_downstream if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate, 'lr': params.downs_en_lr
         },
        {'params': [p for n, p in param_downstream if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': params.downs_en_lr
         }
    ]
    num_train_optimization_steps = len(
        train_loader) // params.gradient_accumulation_steps * args.epoch_num
    optimizer = BertAdam(optimizer_grouped_parameters, warmup=params.warmup_prop, schedule="warmup_cosine",
                         t_total=num_train_optimization_steps, max_grad_norm=params.clip_grad)

    # patience stage
    best_val_f1 = 0.0
    patience_counter = 0

    train_loss = []
    val_loss = []
    f1_score = []
    p_score = []
    r_score = []
   
    for epoch in range(1, args.epoch_num + 1):
        # Run one epoch
        
        logging.info("Epoch {}/{}".format(epoch, args.epoch_num))

        # Train for one epoch on training set
        val_loss_avg, val_loss_avg_seq, val_loss_avg_mat, val_loss_avg_rel = val(model, temp_loader, 
        optimizer, params, ex_params)


        loss_avg, loss_avg_seq, loss_avg_mat, loss_avg_rel = train(model, train_loader, 
        optimizer, params, ex_params)

        train_loss.append(loss_avg)

        val_loss.append(val_loss_avg)
        
        # val(model, val_loader, optimizer, params, ex_params)
        # Evaluate for one epoch on training set and validation set
        # train_metrics = evaluate(args, model, train_loader, params, mark='Train',
        #                          verbose=True)  # Dict['loss', 'f1']


        (val_metrics, _, _ , metrics_ks, 
        metrics_ka, metrics_es, metrics_ea, metrics_d,metrics_k, metrics_a, metrics_s, metrics_e, 
        metrics_di, metrics_ma, TP_all, 
        FP_all, FN_all) = evaluate(model, val_loader, params, ex_params, mark='Val')

        ## add score for precision, recall, and F1   
        val_f1 = val_metrics['f1']
        val_p = val_metrics['precision']
        val_r = val_metrics['recall']
        improve_f1 = val_f1 - best_val_f1
         
        TP = val_metrics['TP']
        FP = val_metrics['FP']
        FN = val_metrics['FN']

        ## add score for precision, recall, and F1
        f1_score.append(val_f1)
        p_score.append(val_p)
        r_score.append(val_r)

        # Save weights of the network
        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self
        optimizer_to_save = optimizer
        utils.save_checkpoint({'epoch': epoch + 1,
                               'model': model_to_save,
                               'optim': optimizer_to_save},
                              is_best=improve_f1 > 0,
                              checkpoint=params.model_dir)
        params.save(params.ex_dir / 'params.json')

        # stop training based params.patience
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter > params.patience_num and epoch > params.min_epoch_num) or epoch == args.epoch_num:
            logging.info("Best val f1: {:05.2f}".format(best_val_f1))
            break

    names = ['knowledge_skills', 'knowledge_area', 'experience_skills', 
          'experience_areas', 'degree_in']
      
    for idx, name in enumerate(names):
      with open('Error_analysis/{}.txt'.format(name),'w', encoding='utf8') as f:
          for _pred in TP_all[idx]:
              for i in _pred:
                f.write(str(i))
                f.write('\n')
          f.write('------TP------')
          for _pred in FP_all[idx]:
              for i in _pred:
                f.write(str(i))
                f.write('\n')
          f.write('------FP------')
          for _pred in FN_all[idx]:
              for i in _pred:
                f.write(str(i))
                f.write('\n')
          f.write('------FN------')

    ####################################################################
    epochs = [i+1 for i in range(len(train_loss))]
    epochs_f1 = [i+1 for i in range(len(f1_score))]

    plt.plot(epochs_f1, f1_score, label= "F1_score")
    plt.plot(epochs_f1, p_score, label= "precision_score")
    plt.plot(epochs_f1, r_score, label= "recall_score")
  
    plt.title('evaluation metrics')
    plt.xlabel('Epochs')
    plt.ylabel('F1/ precision/ recall per epoch')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('./experiments/ex1/f1_validate_1.png', dpi=100)

    ###############################################################

    fig2 = plt.gcf()
    plt.plot(epochs, train_loss, label= "train_loss")
    plt.plot(epochs, val_loss, label= "val_loss")
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Average loss per epoch')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig2.savefig('./experiments/ex1/loss_function_1.png', dpi=100)

    return TP, FP, FN, metrics_ks, metrics_ka, metrics_es, metrics_ea, metrics_d, val_metrics, metrics_k, metrics_a, metrics_s, metrics_e, metrics_di, metrics_ma

if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(args.ex_index, args.corpus_type)
    ex_params = {
        'ensure_corres': args.ensure_corres,
        'ensure_rel': args.ensure_rel,
        'num_negs': args.num_negs,
        'emb_fusion': args.emb_fusion
    }

    if args.multi_gpu:
        params.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        
        n_gpu = torch.cuda.device_count()
        params.n_gpu = n_gpu
    else:
        torch.cuda.set_device(args.device_id)
        print('current device:', torch.cuda.current_device())
        params.n_gpu = n_gpu = 1

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

   
    torch.cuda.manual_seed_all(args.seed)

    # Set the logger
    utils.set_logger(save=True, log_path=os.path.join(
        params.ex_dir, 'train.log'))
    logging.info(f"Model type:")
    logging.info("device: {}".format(params.device))

    logging.info('Load pre-train model weights...')
    bert_config = BertConfig.from_json_file(
        os.path.join(params.bert_model_dir, 'config.json'))
    model = BertForRE.from_pretrained(config=bert_config,
                                      pretrained_model_name_or_path=params.bert_model_dir,
                                      params=params)
    logging.info('-done')

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(args.epoch_num))

    
    # plot confusion matrix for the last epoch
    TP, FP, FN, metrics_ks, metrics_ka, metrics_es, metrics_ea, metrics_d, val_metrics, metrics_k, metrics_a, metrics_s, metrics_e, metrics_di, metrics_ma = train_and_evaluate(model, params, ex_params, args.restore_file)

    TN= 0
    #############################
    """confusion matrix for relation"""
    TPa, TPb, TPc, TPd, TPe = metrics_ks['TP'], metrics_ka['TP'], metrics_es['TP'], metrics_ea['TP'], metrics_d['TP']
    FPa, FPb, FPc, FPd, FPe = metrics_ks['FP'], metrics_ka['FP'], metrics_es['FP'], metrics_ea['FP'], metrics_d['FP']
    FNa, FNb, FNc, FNd, FNe = metrics_ks['FN'], metrics_ka['FN'], metrics_es['FN'], metrics_ea['FN'], metrics_d['FN']
    Micro_average_p = (TPa + TPb + TPc + TPd + TPe) /  (TPa + TPb + TPc + TPd + TPe + FPa +FPb + FPc + FPd + FPe) if (TPa + TPb + TPc + TPd + TPe + FPa +FPb + FPc + FPd + FPe) > 0 else 0
    Micro_average_r = (TPa + TPb + TPc + TPd + TPe) /  (TPa + TPb + TPc + TPd + TPe + FNa +FNb + FNc + FNd + FNe) if  (TPa + TPb + TPc + TPd + TPe + FNa +FNb + FNc + FNd + FNe)> 0 else 0
    Micro_average_f = (metrics_ks['f1'] +metrics_ka['f1'] + metrics_es['f1']+ metrics_ea['f1'] +metrics_d['f1']) / 5

    Macro_average_r = (metrics_ks['recall'] + metrics_ka['recall'] + metrics_es['recall'] + metrics_ea['recall']+ metrics_d['recall'])/ 5
    Macro_average_p = (metrics_ks['precision'] + metrics_ka['precision'] + metrics_es['precision'] + metrics_ea['precision']+ metrics_d['precision'])/ 5
    Macro_average_f = (metrics_ks['f1'] +metrics_ka['f1'] + metrics_es['f1']+ metrics_ea['f1'] +metrics_d['f1']) / 5

    data = [{'Relation': 'knowledge_skills', 'F1_score': metrics_ks['f1'], 'precision': metrics_ks['precision'], 'recall': metrics_ks['recall']},
         {'Relation': 'knowledge_areas',  'F1_score': metrics_ka['f1'], 'precision': metrics_ka['precision'], 'recall': metrics_ka['recall']},
         {'Relation': 'experience_skills',  'F1_score': metrics_es['f1'], 'precision': metrics_es['precision'], 'recall': metrics_es['recall']},
        {'Relation': 'experience_areas',  'F1_score': metrics_ea['f1'], 'precision': metrics_ea['precision'], 'recall': metrics_ea['recall']},
         {'Relation': 'degree_in',  'F1_score': metrics_d['f1'], 'precision': metrics_d['precision'], 'recall': metrics_d['recall'] },
       {'Relation': 'All',  'F1_score': val_metrics['f1'], 'precision': val_metrics['precision'], 'recall': val_metrics['recall'] },
       {'Relation': 'Micro-average',  'F1_score': Micro_average_f, 'precision': Micro_average_p, 'recall': Micro_average_r },
       {'Relation': 'Macro-average',  'F1_score': Macro_average_f , 'precision': Macro_average_p, 'recall': Macro_average_r }]
  
    df_cf = pd.DataFrame.from_dict(data)
  
    df_cf.to_csv('./experiments/ex1/relation_confusion_metrics_1.csv')

    #############################
    """confusion matrix for entity"""

    E_TPa, E_TPb, E_TPc, E_TPd, E_TPe, E_TPf = metrics_k['TP'], metrics_a['TP'], metrics_s['TP'], metrics_e['TP'], metrics_di['TP'], metrics_ma['TP']
    E_FPa, E_FPb, E_FPc, E_FPd, E_FPe, E_FPf = metrics_k['FP'], metrics_a['FP'], metrics_s['FP'], metrics_e['FP'], metrics_di['FP'], metrics_ma['FP']
    E_FNa, E_FNb, E_FNc, E_FNd, E_FNe, E_FNf = metrics_k['FN'], metrics_a['FN'], metrics_s['FN'], metrics_e['FN'], metrics_di['FN'], metrics_ma['FN']
    E_Micro_average_p = (E_TPa + E_TPb + E_TPc + E_TPd + E_TPe+ E_TPf) / (E_TPa + E_TPb + E_TPc + E_TPd + E_TPe + E_TPf + E_FPa +E_FPb + E_FPc + E_FPd + E_FPe+E_FPf) if (E_TPa + E_TPb + E_TPc + E_TPd + E_TPe+E_TPf + E_FPa +E_FPb + E_FPc + E_FPd + E_FPe+E_FPf) > 0 else 0
    E_Micro_average_r = (E_TPa + E_TPb + E_TPc + E_TPd + E_TPe+ E_TPf) / (E_TPa + E_TPb + E_TPc + E_TPd + E_TPe + E_TPf + E_FNa +E_FNb + E_FNc + E_FNd + E_FNe+ E_FNf) if (E_TPa + E_TPb + E_TPc + E_TPd + E_TPe+E_TPf + E_FNa +E_FNb + E_FNc + E_FNd + E_FNe+E_FNf) > 0 else 0
    E_Micro_average_f = (metrics_k['f1'] +metrics_a['f1'] + metrics_s['f1']+ metrics_e['f1'] +metrics_di['f1'] + +metrics_ma['f1']) / 6

    E_Macro_average_r = (metrics_k['recall'] + metrics_a['recall'] + metrics_s['recall'] + metrics_e['recall']+ metrics_di['recall']+metrics_ma['recall'])/ 6
    E_Macro_average_p = (metrics_k['precision'] + metrics_a['precision'] + metrics_s['precision'] + metrics_e['precision']+ metrics_di['precision']+metrics_ma['precision'])/ 6
    E_Macro_average_f = (metrics_k['f1'] +metrics_a['f1'] + metrics_s['f1']+ metrics_e['f1'] +metrics_di['f1']+metrics_ma['f1']) / 6

    data_entity = [{'Entity': 'knowledge', 'F1_score': metrics_k['f1'], 'precision': metrics_k['precision'], 'recall': metrics_k['recall']},
         {'Entity': 'areas',  'F1_score': metrics_a['f1'], 'precision': metrics_a['precision'], 'recall': metrics_a['recall']},
         {'Entity': 'skills', 'F1_score': metrics_s['f1'], 'precision': metrics_s['precision'], 'recall': metrics_s['recall']},
        {'Entity': 'experience',  'F1_score': metrics_e['f1'], 'precision': metrics_e['precision'], 'recall': metrics_e['recall']},
         {'Entity': 'diploam',  'F1_score': metrics_di['f1'], 'precision': metrics_di['precision'], 'recall': metrics_di['recall'] },
          {'Entity': 'major',  'F1_score': metrics_ma['f1'], 'precision': metrics_ma['precision'], 'recall': metrics_ma['recall'] },
       {'Entity': 'Micro-average',  'F1_score': E_Micro_average_f, 'precision': E_Micro_average_p, 'recall': E_Micro_average_r },
       {'Entity': 'Macro-average',  'F1_score': E_Macro_average_f , 'precision': E_Macro_average_p, 'recall': E_Macro_average_r }]
  
    df_cf_e = pd.DataFrame.from_dict(data_entity)
  
    df_cf_e.to_csv('./experiments/ex1/entity_confusion_metrics_1.csv')


    ##############################
    """Visualize the confusion matrixs"""
    
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    cf_matrix1  = np.array([[TP,FP], [FN, 0]])
    cf_matrix2  = np.array([[TPa, FPa], [FNa, 0]])
    cf_matrix3  = np.array([[TPb, FPb], [FNb, 0]])
    cf_matrix4  = np.array([[TPc, FPc], [FNc, 0]])
    cf_matrix5  = np.array([[TPd, FPd], [FNd, 0]])
    cf_matrix6  = np.array([[TPe, FPe], [FNe, 0]])


    fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3, 2, figsize=(8, 10))
    fig.subplots_adjust(wspace=0.01)

    sns.heatmap(cf_matrix1/np.sum(cf_matrix1), annot=True, 
                fmt='.2%', cmap='flare', ax=ax1, linewidths=.5)
    sns.heatmap(cf_matrix2/np.sum(cf_matrix2), annot=True, 
                fmt='.2%', cmap='flare', ax=ax2, linewidths=.5)
    sns.heatmap(cf_matrix3/np.sum(cf_matrix3), annot=True, 
                fmt='.2%', cmap='flare', ax=ax3, linewidths=.5)
    sns.heatmap(cf_matrix4/np.sum(cf_matrix4), annot=True, 
                fmt='.2%', cmap='flare', ax=ax4, linewidths=.5)
    sns.heatmap(cf_matrix5/np.sum(cf_matrix5), annot=True, 
                fmt='.2%', cmap='flare', ax=ax5, linewidths=.5)
    sns.heatmap(cf_matrix6/np.sum(cf_matrix6), annot=True, 
                fmt='.2%', cmap='flare', ax=ax6, linewidths=.5)

    ax1.set_title('All')
    ax1.set_xlabel('actual')
    ax1.set_ylabel('model')
    ax1.xaxis.set_ticklabels(['actual postivie', 'actual negative'])
    ax1.yaxis.set_ticklabels(['predict postivie', 'predict negative'])

    ax2.set_title('Knowledge_skills')
    ax2.set_xlabel('actual')
    ax2.set_ylabel('model')
    ax2.xaxis.set_ticklabels(['actual postivie', 'actual negative'])
    ax2.yaxis.set_ticklabels(['predic postivie', 'predict negative'])

    ax3.set_title('Knowledge_areas')
    ax3.set_xlabel('actual')
    ax3.set_ylabel('model')
    ax3.xaxis.set_ticklabels(['actual postivie', 'actual negative'])
    ax3.yaxis.set_ticklabels(['predict postivie', 'predict negative'])

    ax4.set_title('Experience_skills')
    ax4.set_xlabel('actual')
    ax4.set_ylabel('model')
    ax4.xaxis.set_ticklabels(['actual postivie', 'actual negative'])
    ax4.yaxis.set_ticklabels(['predict postivie', 'predict negative'])

    ax5.set_title('Experience_areas')
    ax5.set_xlabel('actual')
    ax5.set_ylabel('model')
    ax5.xaxis.set_ticklabels(['actual postivie', 'actual negative'])
    ax5.yaxis.set_ticklabels(['predict postivie', 'predict negative'])

    ax6.set_title('Degree_in')
    ax6.set_xlabel('actual')
    ax6.set_ylabel('model')
    ax6.xaxis.set_ticklabels(['actual postivie', 'actual negative'])
    ax6.yaxis.set_ticklabels(['predict postivie', 'predict negative'])

    fig.subplots_adjust(wspace=0.001)

    plt.show()
    plt.draw()
    fig.savefig('./experiments/ex1/relation_confusion_metrics_1.png', dpi=100)
