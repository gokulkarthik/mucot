import pandas as pd
import wandb

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainerCallback

from datasets_local import postprocess_qa_predictions
from utils.metrics import compute_f1_score, computer_jaccard_score



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


class EvaluationCallback(TrainerCallback):

    def __init__(self, dataset, dataset_tokenized, prefix):
        self.dataset = dataset
        self.dataset_tokenized = dataset_tokenized
        self.prefix = prefix
        self.langs = {ex['language'] for ex in self.dataset}
        self.results_jaccard = {'overall':{'step':0, 'score':0}}
        for lang in self.langs:
            self.results_jaccard[lang] = {'step':0, 'score':0}
        self.results_f1 = {'overall':{'step':0, 'score':0}}
        for lang in self.langs:
            self.results_f1[lang] = {'step':0, 'score':0}
        
        if prefix == 'test':
            self.jaccard_test_list = {'overall':[]}
            for lang in self.langs:
                self.jaccard_test_list[lang] = []
            self.f1_test_list = {'overall':[]}
            for lang in self.langs:
                self.f1_test_list[lang] = []

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
            f'{self.prefix}/jaccard.overall': jaccard_score,
            f'{self.prefix}/f1.overall': f1_score,
        }, step=state.global_step)
        if jaccard_score >= self.results_jaccard['overall']['score']:
            self.results_jaccard['overall']['step'] = state.global_step
            self.results_jaccard['overall']['score'] = jaccard_score
        if f1_score >= self.results_f1['overall']['score']:
            self.results_f1['overall']['step'] = state.global_step
            self.results_f1['overall']['score'] = f1_score

        if self.prefix == 'test':
            self.jaccard_test_list['overall'].append(jaccard_score)
            self.f1_test_list['overall'].append(f1_score)

        for language, metrics in result_grouped_dict.items():
            wandb.log({
            f'{self.prefix}/jaccard.{language}': metrics['jaccard_score'],
            f'{self.prefix}/f1.{language}': metrics['f1_score'],
            }, step=state.global_step)
            if  metrics['jaccard_score'] >= self.results_jaccard[language]['score']:
                self.results_jaccard[language]['step'] = state.global_step
                self.results_jaccard[language]['score'] =  metrics['jaccard_score']
            if  metrics['f1_score'] >= self.results_f1[language]['score']:
                self.results_f1[language]['step'] = state.global_step
                self.results_f1[language]['score'] =  metrics['f1_score']

            if self.prefix == 'test':
                self.jaccard_test_list[language].append(metrics['jaccard_score'])
                self.f1_test_list[language].append(metrics['f1_score'])

    def on_train_end(self, args, state, control, **kwargs):
        wandb.run.summary[f'best/{self.prefix}/jaccard'] = self.results_jaccard
        wandb.run.summary[f'best/{self.prefix}/f1'] = self.results_f1
        if self.prefix == 'test':
            wandb.run.summary['test_list_jaccard'] = self.jaccard_test_list
            wandb.run.summary['test_list_f1'] = self.f1_test_list


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
    result_grouped_dict = result_grouped.to_dict(orient='index')

    wandb.run.summary[f'final/{prefix}/jaccard.overvall'] = jaccard_score
    wandb.run.summary[f'final/{prefix}/f1.overvall'] = f1_score

    for language, metrics in result_grouped_dict.items():
        wandb.run.summary[f'final/{prefix}/jaccard.{language}'] = metrics['jaccard_score']
        wandb.run.summary[f'final/{prefix}/f1.{language}'] = metrics['f1_score']

    result.to_csv(f'ckpts/{run_name}/{prefix}_result.csv')
    print(f'{prefix}: Jaccard:{jaccard_score}    F1:{f1_score}')
    print(result_grouped)

    return True


