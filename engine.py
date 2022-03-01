from ast import arg
from random import choice
from re import A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, Trainer, TrainerCallback

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.file_utils import is_sagemaker_mp_enabled, is_apex_available, is_datasets_available

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward
if is_apex_available():
    from apex import amp
if is_datasets_available():
    import datasets

from datasets_local import postprocess_qa_predictions
from utils.metrics import compute_f1_score, computer_jaccard_score


class CustomTrainer(Trainer):

    def __init__(self, model, training_args, **kwargs):
        self.wt_contrastive_loss = kwargs.pop('wt_contrastive_loss')
        self.contrastive_loss_layers = kwargs.pop('contrastive_loss_layers')
        self.agg_for_contrastive = kwargs.pop('agg_for_contrastive')
        self.temperature_for_contrastive = kwargs.pop('temperature_for_contrastive')
        self.max_steps_for_contrastive = kwargs.pop('max_steps_for_contrastive')
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        super(CustomTrainer, self).__init__(model, training_args, **kwargs)
        if self.temperature_for_contrastive < 0:
            self.model.logit_scale = nn.Parameter(torch.ones([], device=self.model.device) * np.log(1 / 0.07))
            # self.model.logit_scale_qn = nn.Parameter(torch.ones([], device=self.model.device) * np.log(1 / 0.07))
            # self.model.logit_scale_con = nn.Parameter(torch.ones([], device=self.model.device) * np.log(1 / 0.07))

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        is_training = False
        if "feature_idx" in inputs.keys():
            is_training = True
            feature_idx = inputs.pop("feature_idx").tolist()
        
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

        if self.wt_contrastive_loss > 0 and is_training: # to avoid running during evaluation
            normalize_embedding = True
            contrastive_loss_method = 'clip' # ['clip']

            # formatting inputs of the pairs
            pair_info_df_batch = self.pair_info_df.loc[feature_idx, :]
            pair_info_df_batch['pair_idx_selected'] = pair_info_df_batch['pair_idx'].apply(lambda x: np.random.choice(x))
            pair_idx_batch = pair_info_df_batch['pair_idx_selected'].values.tolist()
            inputs_pair = self.train_dataset[pair_idx_batch]
            for key in ['overflow_to_sample_mapping', 'local_feature_idx', 'source_idx', 'example_idx', 'pair_idx', 'source_example_idx', 'feature_idx']:
                del inputs_pair[key]
            inputs_pair = {k:torch.Tensor(v).long().to(torch.device(loss.device)) for k, v in inputs_pair.items()}

            # getting the outupts of pairs
            outputs_pair = model(**inputs_pair)
            loss_pair = outputs_pair["loss"]

            contrastive_loss_overall = 0
            for layer_idx in self.contrastive_loss_layers:

                # embed_a = outputs["hidden_states"][layer_idx]  # [bs, seq_len, embed_size]
                # embed_b = outputs["hidden_states"][layer_idx]  # [bs, seq_len, embed_size]
                # outputs["hidden_states"] = [torch.rand(outputs["start_logits"].shape[0], 384, 768, device=loss.device)]*13
                if self.agg_for_contrastive == 'mean':
                    embed_a = torch.mean(outputs["hidden_states"][layer_idx], dim=1) # [bs, embed_size]
                    embed_b = torch.mean(outputs_pair["hidden_states"][layer_idx], dim=1) # [bs, embed_size]
                elif self.agg_for_contrastive == 'max':
                    embed_a, _ = torch.max(outputs["hidden_states"][layer_idx], dim=1) # [bs, embed_size]
                    embed_b, _ = torch.max(outputs_pair["hidden_states"][layer_idx], dim=1) # [bs, embed_size]
                elif self.agg_for_contrastive == 'concat':
                    embed_size = embed_a.shape[2] 
                    embed_a = outputs["hidden_states"][layer_idx].view(-1, embed_size) # [bs*seq_len, embed_size]
                    embed_b = outputs_pair["hidden_states"][layer_idx].view(-1, embed_size) # [bs*seq_len, embed_size]
                elif self.agg_for_contrastive == 'cls':
                    embed_a = outputs["hidden_states"][layer_idx][:, 0, :] # [bs, embed_size]
                    embed_b = outputs_pair["hidden_states"][layer_idx][:, 0, :] # [bs, embed_size]
                else:
                    raise ValueError()
                
                if normalize_embedding:
                    embed_a = F.normalize(embed_a, p=2, dim=1)
                    embed_b = F.normalize(embed_b, p=2, dim=1)

                if self.temperature_for_contrastive < 0:
                    logit_scale = self.model.logit_scale.exp()
                    logits = torch.mm(embed_a, embed_b.t()) * logit_scale
                else:
                    logits = torch.mm(embed_a, embed_b.t()) * self.temperature_for_contrastive

                if contrastive_loss_method == 'clip':
                    labels = torch.arange(logits.shape[0], device=logits.device)
                    a_loss = self.cross_entropy_loss(logits, labels)
                    b_loss = self.cross_entropy_loss(logits.t(), labels)
                    layer_contrastive_loss = (a_loss + b_loss) / 2
                else:
                    raise ValueError()

                contrastive_loss_overall += layer_contrastive_loss
            
            contrastive_loss = contrastive_loss_overall / len(self.contrastive_loss_layers)
            wandb.log({'train/qa_loss': loss}, commit=False)
            wandb.log({'train/contrastive_loss': contrastive_loss}, commit=False)

            # contrastive_loss_overall_qn = 0
            # contrastive_loss_overall_con = 0
            # for layer_idx in self.contrastive_loss_layers:

            #     embed_a = outputs["hidden_states"][layer_idx]  # [bs, seq_len, embed_size]
            #     filter_a_qn = torch.unsqueeze((inputs['token_type_ids']==0).long(), dim=2) # [bs, seq_len, 1]
            #     embed_a_qn = embed_a * filter_a_qn
            #     embed_a_con = embed_a * (1-filter_a_qn)

            #     embed_b = outputs_pair["hidden_states"][layer_idx]  # [bs, seq_len, embed_size]
            #     filter_b_qn = torch.unsqueeze((inputs_pair['token_type_ids']==0).long(), dim=2) # [bs, seq_len, 1]
            #     embed_b_qn = embed_b * filter_b_qn
            #     embed_b_con = embed_b * (1-filter_b_qn)

            #     if self.agg_for_contrastive == 'mean':
            #         embed_a_qn = torch.sum(embed_a_qn, dim=1) / torch.sum(filter_a_qn, dim=1)# [bs, embed_size]
            #         embed_a_con = torch.sum(embed_a_con, dim=1) / torch.sum(1-filter_a_qn, dim=1)# [bs, embed_size]
            #         embed_b_qn = torch.sum(embed_b_qn, dim=1) / torch.sum(filter_b_qn, dim=1)# [bs, embed_size]
            #         embed_b_con = torch.sum(embed_b_con, dim=1) / torch.sum(1-filter_b_qn, dim=1)# [bs, embed_size]
            #     elif self.agg_for_contrastive == 'max':
            #         embed_a_qn, _ = torch.max(embed_a_qn, dim=1) # [bs, embed_size]
            #         embed_a_con, _ = torch.max(embed_a_con, dim=1) # [bs, embed_size]
            #         embed_b_qn, _ = torch.max(embed_b_qn, dim=1) # [bs, embed_size]
            #         embed_b_con, _ = torch.max(embed_b_con, dim=1) # [bs, embed_size]
            #     elif self.agg_for_contrastive == 'cls_sep':
            #         pass
            #     else:
            #         raise ValueError()
                
            #     if normalize_embedding:
            #         embed_a_qn = F.normalize(embed_a_qn, p=2, dim=1)
            #         embed_a_con = F.normalize(embed_a_con, p=2, dim=1)

            #         embed_b_qn = F.normalize(embed_b_qn, p=2, dim=1)
            #         embed_b_con = F.normalize(embed_b_con, p=2, dim=1)

            #     if self.temperature_for_contrastive < 0:
            #         logit_scale_qn = self.model.logit_scale_qn.exp()
            #         logits_qn = torch.mm(embed_a_qn, embed_b_qn.t()) * logit_scale_qn

            #         logit_scale_con = self.model.logit_scale_con.exp()
            #         logits_con = torch.mm(embed_a_con, embed_b_con.t()) * logit_scale_con
            #     else:
            #         logits_qn = torch.mm(embed_a_qn, embed_b_qn.t()) * self.temperature_for_contrastive

            #         logits_con = torch.mm(embed_a_con, embed_b_con.t()) * self.temperature_for_contrastive

            #     if contrastive_loss_method == 'clip':
            #         labels = torch.arange(logits_qn.shape[0], device=logits_qn.device)

            #         a_loss_qn = self.cross_entropy_loss(logits_qn, labels)
            #         b_loss_qn = self.cross_entropy_loss(logits_qn.t(), labels)
            #         layer_contrastive_loss_qn = (a_loss_qn + b_loss_qn) / 2

            #         a_loss_con = self.cross_entropy_loss(logits_con, labels)
            #         b_loss_con = self.cross_entropy_loss(logits_con.t(), labels)
            #         layer_contrastive_loss_con = (a_loss_con + b_loss_con) / 2
            #     else:
            #         raise ValueError()

            #     contrastive_loss_overall_qn += layer_contrastive_loss_qn
            #     contrastive_loss_overall_con += layer_contrastive_loss_con
            
            # contrastive_loss_qn = contrastive_loss_overall_qn / len(self.contrastive_loss_layers)
            # contrastive_loss_con = contrastive_loss_overall_con / len(self.contrastive_loss_layers)
            # contrastive_loss = (contrastive_loss_qn + contrastive_loss_con) / 2
            # wandb.log({'train/qa_loss': loss}, commit=False)
            # wandb.log({'train/qa_loss_pair': loss_pair}, commit=False)
            # wandb.log({'train/contrastive_loss_qn': contrastive_loss_qn}, commit=False)
            # wandb.log({'train/contrastive_loss_con': contrastive_loss_con}, commit=False)
            # wandb.log({'train/contrastive_loss': contrastive_loss}, commit=False)


            if self.state.global_step < self.max_steps_for_contrastive:
                loss = loss + self.wt_contrastive_loss*contrastive_loss #+ loss_pair

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            #train_dataset = self._remove_unused_columns(train_dataset, description="training")
            self.pair_info_df = train_dataset.to_pandas()[['feature_idx', 'pair_idx']].set_index('feature_idx')
            train_dataset = train_dataset.remove_columns(['overflow_to_sample_mapping', 'local_feature_idx', 'source_idx', 'example_idx', 'pair_idx', 'source_example_idx'])

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )



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

        model.config.output_hidden_states = False # To handle out of memory
        eval_trainer = Trainer(model)
        predictions_raw = eval_trainer.predict(self.dataset_tokenized)
        model.config.output_hidden_states = True # To handle out of memory: undo for training

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
    config = AutoConfig.from_pretrained(args.model_ckpt, output_hidden_states=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_ckpt, config=config)
    return model

def evaluate_model(model, tokenizer, dataset, dataset_tokenized, prefix, run_name):
    model.config.output_hidden_states = False # To handle out of memory
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


