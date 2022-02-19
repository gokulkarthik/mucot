import argparse

import pandas as pd
import wandb

from transformers import EarlyStoppingCallback, default_data_collator
from transformers import TrainingArguments, Trainer

from datasets_local import load_dataset, postprocess_qa_predictions
from engine import CustomTrainer, create_tokenizer, create_model
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
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--doc_stride', type=int, default=128)

    # model parameters
    parser.add_argument('--model_ckpt', type=str, help='Local path or huggingface url', required=True)

    # training parameters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=is_even, default=4, help='Batch size must be an even number')
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # other parameters
    parser.add_argument('--eval', type=str2bool, default=False, help='Perform evaluation only')


    return parser


def main(args):

    tokenizer = create_tokenizer(args)
    dataset_train, dataset_train_tokenized = load_dataset(args=args, split='train', mode='train', tokenizer=tokenizer)
    dataset_test, dataset_test_tokenized = load_dataset(args=args, split='test', mode='train', tokenizer=tokenizer)
    model = create_model(args)

    wandb.init(project='mlqa')
    wandb.config.update(args)

    training_args = TrainingArguments(
        f"ckpts",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True, 
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        seed=0,
        report_to='wandb',
        run_name='mlqa'
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset_train_tokenized,
        eval_dataset=dataset_test_tokenized,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = 2)]
    )
        
    if not args.eval:
        trainer.train()

    dataset_train, dataset_train_tokenized = load_dataset(args=args, split='train', mode='test', tokenizer=tokenizer)

    train_predictions_raw = trainer.predict(dataset_train_tokenized)
    train_predictions = postprocess_qa_predictions(dataset_train, dataset_train_tokenized, 
      train_predictions_raw.predictions, tokenizer)
    train_actuals = [{"id": ex["id"], "language": ex["language"], "answer": ex["answers"]['text'][0]} for ex in dataset_train]

    result_train = pd.DataFrame(train_actuals)
    result_train['prediction'] = result_train['id'].apply(lambda r: train_predictions[r])
    result_train['jaccard_score'] = result_train[['answer', 'prediction']].apply(computer_jaccard_score, axis=1)
    result_train['f1_score'] = result_train[['answer', 'prediction']].apply(compute_f1_score, axis=1)
    result_train.to_csv('temp/result_train.csv')
    train_jaccard_score, train_f1_score = result_train['jaccard_score'].mean(), result_train['f1_score'].mean()
    print(f'Train: Jaccard:{train_jaccard_score}    F1:{train_f1_score}')
    result_train_grouped = result_train.groupby('language')[['jaccard_score', 'f1_score']].mean()
    print(result_train_grouped)

    wandb.log({
        'train_jaccard_score': train_jaccard_score,
        'train_f1_score': train_f1_score,
        'train_grouped': wandb.Table(dataframe=result_train_grouped)
    })

    dataset_test, dataset_test_tokenized = load_dataset(args=args, split='test', mode='test', tokenizer=tokenizer)

    test_predictions_raw = trainer.predict(dataset_test_tokenized)
    test_predictions = postprocess_qa_predictions(dataset_test, dataset_test_tokenized, 
        test_predictions_raw.predictions, tokenizer)
    test_actuals = [{"id": ex["id"], "language": ex["language"], "answer": ex["answers"]['text'][0]} for ex in dataset_test]

    result_test = pd.DataFrame(test_actuals)
    result_test['prediction'] = result_test['id'].apply(lambda r: test_predictions[r])
    result_test['jaccard_score'] = result_test[['answer', 'prediction']].apply(computer_jaccard_score, axis=1)
    result_test['f1_score'] = result_test[['answer', 'prediction']].apply(compute_f1_score, axis=1)
    result_test.to_csv('temp/result_test.csv')
    test_jaccard_score, test_f1_score = result_test['jaccard_score'].mean(), result_test['f1_score'].mean()
    print(f'Test: Jaccard:{test_jaccard_score}    F1:{test_f1_score}')
    result_test_grouped = result_test.groupby('language')[['jaccard_score', 'f1_score']].mean()
    print(result_test_grouped)

    wandb.log({
        'test_jaccard_score': test_jaccard_score,
        'test_f1_score': test_f1_score,
        'test_grouped': wandb.Table(dataframe=result_test_grouped),
    })
    

if __name__ == '__main__':

    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)