import pandas as pd
import wandb

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainerCallback

from datasets_local import postprocess_qa_predictions
from utils.metrics import compute_f1_score, computer_jaccard_score


class EvaluationCallback(TrainerCallback):

    def __init__(self, dataset, dataset_tokenized, prefix):
        self.dataset = dataset
        self.dataset_tokenized = dataset_tokenized
        self.prefix = prefix

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.pop('model')
        tokenizer = kwargs.pop('tokenizer')
        eval_trainer = Trainer(model)

        predictions_raw = eval_trainer.predict(self.dataset_tokenized)
        predictions = postprocess_qa_predictions(self.dataset, self.dataset_tokenized, 
            predictions_raw.predictions, tokenizer)
        actuals = [{"id": ex["id"], "language": ex["language"], "answer": ex["answers"]['text'][0]} for ex in self.dataset]
        
        result = pd.DataFrame(actuals)
        result['prediction'] = result['id'].apply(lambda r: predictions[r])
        result['jaccard_score'] = result[['answer', 'prediction']].apply(computer_jaccard_score, axis=1)
        result['f1_score'] = result[['answer', 'prediction']].apply(compute_f1_score, axis=1)
        jaccard_score, f1_score = result['jaccard_score'].mean(), result['f1_score'].mean()
        result_grouped = result.groupby('language')[['jaccard_score', 'f1_score']].mean()
        result_grouped_dict = result_grouped.to_dict(orient='index')

        wandb.log({
            f'{self.prefix}_jaccard_score': jaccard_score,
            f'{self.prefix}_f1_score': f1_score,
        })

        for language, metrics in result_grouped_dict.items():
            print(metrics)
            wandb.log({
            f'{self.prefix}_jaccard_score_{language}': metrics['jaccard_score'],
            f'{self.prefix}_f1_score_{language}': metrics['f1_score'],
            })



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

def create_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    return tokenizer

def create_model(args):
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_ckpt)
    return model

def evaluate_model(model, tokenizer, dataset, dataset_tokenized, prefix, run_name):
    eval_trainer = Trainer(model)

    predictions_raw = eval_trainer.predict(dataset_tokenized)
    predictions = postprocess_qa_predictions(dataset, dataset_tokenized, 
        predictions_raw.predictions, tokenizer)
    actuals = [{"id": ex["id"], "language": ex["language"], "answer": ex["answers"]['text'][0]} for ex in dataset]
    
    result = pd.DataFrame(actuals)
    result['prediction'] = result['id'].apply(lambda r: predictions[r])
    result['jaccard_score'] = result[['answer', 'prediction']].apply(computer_jaccard_score, axis=1)
    result['f1_score'] = result[['answer', 'prediction']].apply(compute_f1_score, axis=1)
    jaccard_score, f1_score = result['jaccard_score'].mean(), result['f1_score'].mean()
    result_grouped = result.groupby('language')[['jaccard_score', 'f1_score']].mean()
    result_grouped['language'] = result_grouped.index


    wandb.log({
        f'final_{prefix}_jaccard_score': jaccard_score,
        f'final_{prefix}_f1_score': f1_score,
        f'final_{prefix}_grouped': wandb.Table(dataframe=result_grouped),
    })

    result.to_csv(f'ckpts/{run_name}/{prefix}_result.csv')
    print(f'{prefix}: Jaccard:{jaccard_score}    F1:{f1_score}')
    print(result_grouped)

    return True


