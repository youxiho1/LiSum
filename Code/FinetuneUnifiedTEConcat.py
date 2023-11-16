import torch
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers import AutoConfig
from transformers import TrainingArguments
from utils.MyTrainerLite import MyTrainerLite
from dataset.TldrLegalDataset.TldrLegalDataset import TldrLegalDataset, MyDataCollator
from model.BartTEUnifiedConcatArch import BartTEUnifiedConcatArch
from config.config import config
import argparse
from datetime import datetime
import json
import torchmetrics
from rouge import Rouge
from utils.SetSeedHelper import set_seed
import shutil
import os

def calculate_average_rouge(rouge_list, kfold):
    avg_rouge = {'r':0.0, 'p':0.0, 'f':0.0}
    for rouge in rouge_list:
        avg_rouge['r'] += rouge['r']
        avg_rouge['p'] += rouge['p']
        avg_rouge['f'] += rouge['f']
    avg_rouge['r'] /= kfold
    avg_rouge['p'] /= kfold
    avg_rouge['f'] /= kfold
    return avg_rouge

if __name__ == '__main__':
    time_mark = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='only_a_test_TE')
    parser.add_argument('--init_mode', default='brief')
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--classification_loss_weight', default=1)
    parser.add_argument('--summarization_loss_weight', default=1)
    parser.add_argument('--model_size', default='base')
    parser.add_argument('--average_method', default='micro')
    parser.add_argument('--result_dir', default='./LicenseComprehension/evaluation_results')
    parser.add_argument('--trans', default='no')
    parser.add_argument('--concat_mode', default='concat')
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--seed', default=1)
    args = parser.parse_args()
    args.learning_rate = float(args.learning_rate)
    print(args.learning_rate)
    args.classification_loss_weight = float(args.classification_loss_weight)
    args.summarization_loss_weight = float(args.summarization_loss_weight)
    args.dropout = float(args.dropout)
    args.seed = int(args.seed)
    set_seed(args.seed)
    
    rouge1_list = []
    rouge2_list = []
    rougel_list = []

    for kfold in range(config.K_FOLDS):
        if args.trans == 'no':
            input_path = 'kfold/licenses_train_set_kfold-'+str(kfold)+'.json'
        else:
            # back trans
            input_path = 'kfold/'
        output_dir = args.model_name+'_fold'+str(kfold)+'/'

        if args.model_size == 'large':
            tokenizer_path = "facebook/bart-large-cnn"
            bart_model_path = 'facebook/bart-large-cnn'
        else:
            tokenizer_path = "ainize/bart-base-cnn"
            bart_model_path = 'ainize/bart-base-cnn'
        

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        raw_dataset = TldrLegalDataset(tokenizer, config.MAX_FULLTEXT_LENGTH, config.MAX_SUMMARY_LENGTH, input_path, config.NUM_CLASSIFIERS)
        train_size = int(config.TRAIN_EVAL_SPLIT_RATIO * len(raw_dataset))
        eval_size = len(raw_dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(raw_dataset, [train_size, eval_size])
        data_collator = MyDataCollator(tokenizer)
        num_labels = len(config.NAME2LABEL)
        bart_config = AutoConfig.from_pretrained(bart_model_path)
        additional_configs = {
            'classifiers_num': config.NUM_CLASSIFIERS,
            'classification_output_dim': num_labels,
            'classification_loss_weight': args.classification_loss_weight,
            'summarization_loss_weight': args.summarization_loss_weight,
            'concat_mode': args.concat_mode,
            'dropout': args.dropout
        }
        bart_config.update(additional_configs)
        
        model = BartTEUnifiedConcatArch.from_pretrained(bart_model_path, config=bart_config)
        
        if args.init_mode == 'brief':
            # initialize attention matrix by encoder (knowledge injection)
            print('init by brief knowledge')
            model.init_attention_matrix_by_task(tokenizer, config.MAX_FULLTEXT_LENGTH)
        elif args.init_mode == 'desp':
            print('init by desp')
            model.init_attention_matrix_by_desp(tokenizer, config.MAX_FULLTEXT_LENGTH)
        model = model.cuda()

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=30,
            learning_rate=args.learning_rate,
            logging_strategy='epoch',
            load_best_model_at_end=True
            # warmup_ratio=0.1,
        )

        trainer = MyTrainerLite(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        trainer.train()
        best_model_path = trainer.get_best_model_path()
        print(best_model_path)
        
        # load best model & evaluate summarization
        filepath = 'kfold/licenses_test_set_kfold-'+str(kfold)+'.json'
        base_path = './LicenseComprehension/results/' + args.model_name+'_fold'+str(kfold)+'/'
        if filepath.__contains__('train_set'):
            output_path = base_path + 'train_set/'
        else:
            output_path = base_path + 'test_set/'
        
        os.makedirs(output_path, exist_ok=True)
        tldr_dataset = TldrLegalDataset(tokenizer, config.MAX_FULLTEXT_LENGTH, config.MAX_SUMMARY_LENGTH, filepath, config.NUM_CLASSIFIERS)
        _config = AutoConfig.from_pretrained(best_model_path)
        
        model = BartTEUnifiedConcatArch.from_pretrained(best_model_path, config=_config)
        model = model.cuda()

        hyps = []
        refs = []
        
        model.eval()
        for index in range(tldr_dataset.__len__()):
            # pred
            info = tldr_dataset.__getinfo__(index)
            license_title = info['license_title']
            fulltext = info['fulltext']
            quick_summary = info['quick_summary']
            refs.append(quick_summary)
            inputs = tokenizer([fulltext], max_length=config.MAX_FULLTEXT_LENGTH, return_tensors='pt')
            input_ids = inputs["input_ids"].cuda()
            attention_mask = inputs["attention_mask"].cuda()
            summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, min_length=3, max_length=200)
            output = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            hyps.append(output)
            # save results to file
            path = output_path + str(index) + '.txt'
            with open(path, 'w', encoding='utf-8') as f:
                lines = ['License Title\n', license_title,
                        '\n\nGenerated Summary\n', output,
                        '\n\nQuick Summary\n', quick_summary,
                        '\n\nFull Text\n', fulltext]
                f.writelines(lines)
            f.close()

        # calculate rouge metric
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=True, ignore_empty=True)
        print(best_model_path)
        print(scores)
        rouge1_list.append(scores['rouge-1'])
        rouge2_list.append(scores['rouge-2'])
        rougel_list.append(scores['rouge-l'])
        
        # evaluate classification     
        test_set_path = 'kfold/licenses_test_set_kfold-'+str(kfold)+'.json'
        test_dataset = TldrLegalDataset(tokenizer, config.MAX_FULLTEXT_LENGTH, config.MAX_SUMMARY_LENGTH, test_set_path, config.NUM_CLASSIFIERS)
        result = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        num_classifiers = config.NUM_CLASSIFIERS
        # predict result
        outputs = torch.zeros(test_dataset.__len__(), num_classifiers)
        model.eval()
        for index in range(test_dataset.__len__()):
            with torch.no_grad():
                one = test_dataset.__getitem__(index)
                summary_ids = model.generate(input_ids=torch.tensor([one['input_ids']]).long().cuda(), attention_mask=torch.tensor([one['attention_mask']]).cuda(), min_length=3, max_length=200)
                
                
                logits = model.forward(input_ids=torch.tensor([one['input_ids']]).long().cuda(),
                                            attention_mask=torch.tensor([one['attention_mask']]).cuda(), labels=summary_ids)['classification_logits']
                                            # attention_mask=torch.tensor([one['attention_mask']]).cuda())['classification_logits']

                predicted_class_ids = logits.argmax(dim=2).permute(1,0)
                # predicted_class_ids.shape = c * b * h -> c * b -> b * c
                outputs[index] = predicted_class_ids[0]
        # print(outputs)
        # outputs = torch.tensor(outputs)
        
        # ground truth
        ground_truth_actions = []
        for one in test_dataset:
            ground_truth_actions.append(one['actions'].tolist())
        ground_truth_actions = torch.tensor(ground_truth_actions)
        # print(ground_truth_actions)

        accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=4, average='micro', top_k=1)
        precision_metric = torchmetrics.Precision(task='multiclass', num_classes=4, average='micro', top_k=1)
        recall_metric = torchmetrics.Recall(task='multiclass', num_classes=4, average='micro', top_k=1)
        f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=4, average='micro', top_k=1)

        for col in range(num_classifiers):
            acc = accuracy_metric(ground_truth_actions[:, col], outputs[:, col])

            result['accuracy'].append(acc.item())

            prec = precision_metric(ground_truth_actions[:, col], outputs[:, col])
            result['precision'].append(prec.item())

            rec = recall_metric(ground_truth_actions[:, col], outputs[:, col])
            result['recall'].append(rec.item())

            f1 = f1_metric(ground_truth_actions[:, col], outputs[:, col])
            result['f1'].append(f1.item())


        result_dir = args.result_dir
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        model_path = args.model_name+'_fold'+str(kfold)
        output_result_json_file_path = result_dir + '/{}-{}-micro.json'.format(model_path.split('/')[-1],
                                                                            time_mark)
        with open(output_result_json_file_path, 'w', encoding='utf-8') as f:
            f.write(best_model_path + '\n')
            json.dump({'model_path': model_path, 'result': result}, f)
               
        result = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=4, average='macro', top_k=1)
        precision_metric = torchmetrics.Precision(task='multiclass', num_classes=4, average='macro', top_k=1)
        recall_metric = torchmetrics.Recall(task='multiclass', num_classes=4, average='macro', top_k=1)
        f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=4, average='macro', top_k=1)

        for col in range(num_classifiers):
            acc = accuracy_metric(ground_truth_actions[:, col], outputs[:, col])

            result['accuracy'].append(acc.item())

            prec = precision_metric(ground_truth_actions[:, col], outputs[:, col])
            result['precision'].append(prec.item())

            rec = recall_metric(ground_truth_actions[:, col], outputs[:, col])
            result['recall'].append(rec.item())

            f1 = f1_metric(ground_truth_actions[:, col], outputs[:, col])
            result['f1'].append(f1.item())


        result_dir = args.result_dir
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        model_path = args.model_name+'_fold'+str(kfold)
        output_result_json_file_path = result_dir + '/{}-{}-macro.json'.format(model_path.split('/')[-1],
                                                                            time_mark)
        with open(output_result_json_file_path, 'w', encoding='utf-8') as f:
            f.write(best_model_path + '\n')
            json.dump({'model_path': model_path, 'result': result}, f)

    # calculate average summarization metric
    avg_rouge1 = calculate_average_rouge(rouge1_list, config.K_FOLDS)
    avg_rouge2 = calculate_average_rouge(rouge2_list, config.K_FOLDS)
    avg_rougel = calculate_average_rouge(rougel_list, config.K_FOLDS)
    # write rouge into file
    result_path = './LicenseComprehension/results/' + args.model_name + '.txt'
    with open(result_path, 'w') as f:
        f.write(args.model_name + '\n')
        f.write(str(avg_rouge1) + '\n')
        f.write(str(avg_rouge2) + '\n')
        f.write(str(avg_rougel) + '\n')
    # calculate average classification metric
    
    
    
