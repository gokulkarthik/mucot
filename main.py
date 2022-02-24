import argparse
from operator import contains

import pandas as pd
import wandb

from transformers import EarlyStoppingCallback, default_data_collator
from transformers import TrainingArguments, Trainer

from datasets_local import load_dataset, postprocess_qa_predictions, add_pair_idx_column
from engine import CustomTrainer, EvaluationCallback, create_tokenizer, create_model, evaluate_model
from utils.metrics import compute_f1_score, computer_jaccard_score

def str2bool(v):
    """
    src: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def is_even(v):
    if isinstance(v, int):
        if v % 2 == 0:
            return True
    return False

def get_arg_parser():

    parser = argparse.ArgumentParser(description='Training and evaluation script for multilingual question answering')

    # dataset parameters
    parser.add_argument('--dataset', default='chaii', choices=['chaii'])
    parser.add_argument('--langs', choices=['hi', 'ta', 'en^', 'bn^', 'hi^', 'mr^', 'ml^', 'ta^', 'te^'], nargs='+')
    parser.add_argument('--min_langs', type=int, default=1)
    parser.add_argument('--langs_for_min_langs_filter', choices=['hi', 'ta', 'en^', 'bn^', 'hi^', 'mr^', 'ml^', 'ta^', 'te^'], nargs='+')
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--doc_stride', type=int, default=128)

    # model parameters
    parser.add_argument('--model_name', type=str, default="", choices=['mbert', 'mbert-squad', 'xlmroberta', 'xlmroberta-squad', 'distillmbert', 'muril', 'indic-bert'], required=False)
    parser.add_argument('--model_ckpt', type=str, default="", help='Local path or huggingface url', required=False)

    # training parameters
    parser.add_argument('--wt_contrastive_loss', type=float, default=0.0)
    parser.add_argument('--contrastive_loss_layers', nargs='+')
    parser.add_argument('--agg_for_contrastive', type=str, default="mean", choices=['mean', 'max', 'concat'], required=False)
    parser.add_argument('--max_steps_for_contrastive', type=int, default=5000)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--logging_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--train_batch_size', type=int, default=4, help='Batch size must be an even number')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='Batch size must be an even number')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # other parameters
    parser.add_argument('--eval', type=str2bool, default=False, help='Perform evaluation only')
    parser.add_argument('--debug', type=str2bool, default=False, help='Set to debug mode')
    parser.add_argument('--max_rows', type=int, default=-1, help='Used only in debug mode')

    return parser

def main(args):

    tokenizer = create_tokenizer(args)
    dataset_train, dataset_train_tokenized = load_dataset(args=args, split='train', mode='train', tokenizer=tokenizer)
    dataset_val, dataset_val_tokenized = load_dataset(args=args, split='val', mode='train', tokenizer=tokenizer)
    model = create_model(args)

    # for contrastive training
    dataset_train_tokenized = add_pair_idx_column(dataset_train, dataset_train_tokenized)

    # for evaluation callback
    #dataset_train_4eval, dataset_train_tokenized_4eval = load_dataset(args=args, split='train', mode='eval', tokenizer=tokenizer)
    dataset_val_4eval, dataset_val_tokenized_4eval = load_dataset(args=args, split='val', mode='eval', tokenizer=tokenizer)
    dataset_test_4eval, dataset_test_tokenized_4eval = load_dataset(args=args, split='test', mode='eval', tokenizer=tokenizer)
    
    if args.debug:
        wandb.init(project='mlqa', mode='disabled')
    else:
        wandb.init(project='mlqa', mode='online')
    wandb.config.update(args)
    wandb.config.update({
        'num_params': sum(p.numel() for p in  model.parameters()),
        'num_train_examples': len(dataset_train),
        'num_train_features': len(dataset_train_tokenized)
        })
    run_name = wandb.run.name

    training_args = TrainingArguments(
        f"ckpts/{run_name}", 
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_epochs,
        max_steps = args.max_steps,
        seed=0,
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to='wandb',
        run_name='mlqa'
    )

    trainer = CustomTrainer(
        model,
        training_args,
        train_dataset=dataset_train_tokenized,
        eval_dataset=dataset_val_tokenized,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        callbacks = [
#            EarlyStoppingCallback(early_stopping_patience = 5), 
#            EvaluationCallback(dataset=dataset_train_4eval, dataset_tokenized=dataset_train_tokenized_4eval, prefix='train'),
            EvaluationCallback(dataset=dataset_val_4eval, dataset_tokenized=dataset_val_tokenized_4eval, prefix='val'),
            EvaluationCallback(dataset=dataset_test_4eval, dataset_tokenized=dataset_test_tokenized_4eval, prefix='test')
        ],
        wt_contrastive_loss = args.wt_contrastive_loss,
        contrastive_loss_layers = [int(x) for x in args.contrastive_loss_layers],
        agg_for_contrastive = args.agg_for_contrastive,
        max_steps_for_contrastive = args.max_steps_for_contrastive
    )
        
    if not args.eval:
        # wandb.summary.best - [val, test] split metric based on [corresponding best scores] in the [corresponding] split
        trainer.train()

    # Final Evaluation 
    # wandb.summary.final - [train, val, test] split metrics based on [overall eval loss] on [val] split
    wandb.summary['final/step'] = int(trainer.state.best_model_checkpoint.rsplit('-', 1)[-1])
    #evaluate_model(model, tokenizer, dataset_train_4eval, dataset_train_tokenized_4eval, prefix='train', run_name=run_name)
    evaluate_model(model, tokenizer, dataset_val_4eval, dataset_val_tokenized_4eval, prefix='val', run_name=run_name)
    evaluate_model(model, tokenizer, dataset_test_4eval, dataset_test_tokenized_4eval, prefix='test', run_name=run_name)

    # wandb.summary.result - [test] split metric based on [corresponding best scores] in the [val] split
    groups = wandb.summary['best/val/jaccard'].keys() # overall, hi, ta
    jaccard_result = {}
    f1_result = {}
    for group in groups:
        best_jaccard_step = wandb.summary[f'best/val/jaccard'][group]['step']
        jaccard_result[group] = wandb.summary['test_list_jaccard'][group][(best_jaccard_step//trainer.args.eval_steps)-1]
        best_f1_step = wandb.summary[f'best/val/f1'][group]['step']
        f1_result[group] = wandb.summary['test_list_f1'][group][(best_f1_step//trainer.args.eval_steps)-1]
    wandb.summary['result'] = {
        'jaccard': jaccard_result,
        'f1': f1_result
    }


if __name__ == '__main__':

    parser = get_arg_parser()
    args = parser.parse_args()
    if args.debug:
        args.max_steps = 50
        args.logging_steps = 10
        args.eval_steps = 10
        args.save_steps = 10
        args.max_rows = 100
    
    model_name_to_ckpt = {
        'mbert': 'bert-base-multilingual-cased',
        'mbert-squad': 'salti/bert-base-multilingual-cased-finetuned-squad',
        'xlmroberta': 'xlm-roberta-base',
        'xlmroberta-squad': 'deepset/xlm-roberta-base-squad2',
        'distillmbert': 'distilbert-base-multilingual-cased',
        'muril': 'google/muril-base-cased',
        'indicbert': 'ai4bharat/indic-bert'
    }
    if args.model_name:
        args.model_ckpt = model_name_to_ckpt[args.model_name]
        
    main(args)