# 3-*- coding: utf-8 -*-
"""
1. Train model once the best hyperparameters from final_configs.json that were identified using classification_wandb.py
2. Get predictions from the trained model with boolen flag

"""

import time
import sys
import shutil
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import EarlyStoppingCallback, TrainerCallback

import evaluate
# from evaluater import evaluation_display

hyperparams = ['MODEL_CHECKPOINT','BATCH_SIZE','LEARNING_RATE','EPOCHS','MAX_LEN']

# class PrintClassificationCallback(TrainerCallback):
#     def on_evaluate(self, args, state, control, logs=None, **kwargs):
#         print("Called after evaluation phase")

class ModelTrainer(object):

    def __init__(self, run_config, train_dataset, valid_dataset, test_dataset,
                 tokenizer, savepath, cachepath, num_seed, lang_to_train='all'):     

        self._run_config = run_config
        if not all(k in self._run_config.keys() for k in hyperparams):
            sys.exit(f"provide all required hyperparams: {hyperparams}")
            
        self.model_checkpoint = self._run_config['MODEL_CHECKPOINT']
        self.model_name = self.model_checkpoint.split('/')[-1]
    
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset
        self._train_dataset.cleanup_cache_files()

        self._tokenizer = tokenizer
        
        self._savepath = savepath
        self.modelpath = self._savepath.joinpath('models')
        self.modelpath.mkdir(parents=True, exist_ok=True)
        self._cachepath = cachepath

        self._num_seed = num_seed

        self._lang_to_train = lang_to_train
        # self._target_names = self._run_config['TARGET_NAMES']
        self.num_labels = len(self._run_config['TARGET_NAMES'])
        print(f"{self.num_labels} classes in {self._lang_to_train} language")

    def get_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint,
                                                                   num_labels = self.num_labels,
                                                                   cache_dir = self._cachepath,
                                                                #    output_attentions=False,
                                                                #    output_hidden_states=False,
                                                                #    ignore_mismatched_sizes=True,
                                                                )
        # print(model)
        return model
        
    def compute_metrics(self, eval_pred, eval_metric="f1"):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # evaluation_display(labels, predictions, self.num_labels, self._target_names) # print only for now
        metric = evaluate.load(eval_metric)
        return metric.compute(predictions=predictions, references=labels, average="macro")

    def train_eval(self, get_pred=False, metric_name="f1", hyperparam_search=False): #np.random.randint(1000)):
        print(f"seed used for training: {self._num_seed}")
        
        out_dir = self.modelpath.joinpath(f"{self.model_name}-{self._lang_to_train}-finetuned")
        print(f"Model to be saved in {out_dir}")
        log_dir = self.modelpath.parent.joinpath('logs').joinpath(f"{self.model_name}-{self._lang_to_train}-finetuned")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        config = SimpleNamespace(**{i.lower():j for i,j in self._run_config.items() if i in hyperparams})
        print(f"\nFinal configurations for training the model:\n{config}")
        
        # attributes to customize the training
        args = TrainingArguments(
            seed=self._num_seed,
            save_total_limit=2,
            output_dir=str(out_dir),
            overwrite_output_dir = True,
            
            learning_rate = config.learning_rate,
            per_device_train_batch_size = config.batch_size,
            per_device_eval_batch_size = config.batch_size,
            num_train_epochs = config.epochs,
            # weight_decay = config.weight_decay,
            
            evaluation_strategy = "epoch",
            save_strategy = "epoch",   
            logging_strategy = 'epoch',
            
            # logging_steps= 1,
            eval_accumulation_steps = 1,
            
            metric_for_best_model = metric_name,
            load_best_model_at_end = True,
            
            push_to_hub = False, # push the model to the Hub regularly during training
            # report_to='wandb',  # turn on wandb logging
            )

        # https://huggingface.co/docs/transformers/main_classes/trainer#trainer
        # https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/trainer.py#L231
        trainer = Trainer(
            model_init = self.get_model,
            args = args,
            train_dataset = self._train_dataset,
            eval_dataset = self._valid_dataset,
            tokenizer = self._tokenizer,
            compute_metrics = self.compute_metrics,
            callbacks = [EarlyStoppingCallback(3, 0.0)]
            )
        
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary(device=None, abbreviated=True))
        
        # try later setup hyperparam here instead of wandb
        # # https://huggingface.co/docs/transformers/hpo_train
        # if hyperparam_search:
        #     best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
        #     print(f"best run for {self._model_checkpoint} is {best_run.hyperparameters.items()}")
        #     for n, v in best_run.hyperparameters.items():
        #         setattr(trainer.args, n, v)

        trainer.train()
        trainer.evaluate()
        trainer.save_model("./model")
        print(f"free space by deleting: {out_dir}")
        shutil.rmtree(out_dir, ignore_errors=True)
        
        if get_pred:
            print(f"\nGetting Predictions on Test dataset")
            logits, labels, metrics = trainer.predict(self._test_dataset)
            predictions = np.argmax(logits, axis=-1)
            return trainer, (labels, predictions)   
        else:
            return trainer