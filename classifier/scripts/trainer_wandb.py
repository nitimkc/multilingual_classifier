import time
import shutil

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import EarlyStoppingCallback, TrainerCallback

import evaluate
# from evaluater import evaluation_display

import wandb

def compute_metrics(eval_pred, eval_metric="f1"):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load(eval_metric)
    return metric.compute(predictions=predictions, references=labels, average="macro")
    # precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    # acc = accuracy_score(labels, predictions)
    # return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
    # accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    # f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    # return {"accuracy": accuracy, "f1": f1}

class ModelTrainer(object):

    def __init__(self, model_checkpoint, train_dataset, valid_dataset, test_dataset, 
                 tokenizer, savepath, cachepath, target_names=None, lang_to_train='all'):     
        
        self._model_checkpoint = model_checkpoint
        self.model_name = self._model_checkpoint.split("/")[-1]
        
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        
        self._tokenizer = tokenizer
        
        self._cachepath = cachepath#.joinpath('.cache')
        self._savepath = savepath
        self.modelpath = self._savepath.joinpath('models')
        self.modelpath.mkdir(parents=True, exist_ok=True)
        
        
        self._target_names = target_names
        self.num_labels = len(self._target_names)
        
        self._lang_to_train = lang_to_train
        
    def get_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self._model_checkpoint,
                                                                   num_labels=self.num_labels,
                                                                   cache_dir=self._cachepath,
                                                                #    output_attentions=False,
                                                                #    output_hidden_states=False,
                                                                #    ignore_mismatched_sizes=True,
                                                                )
        # wandb.watch(model, criterion, log="all", log_freq=10)
        # print(model.config)
        return model
    
    def train_eval(self, config=None, metric_name="f1"):
        out_dir = self.modelpath.joinpath(f"{self.model_name}-{self._lang_to_train}-finetuned")
        log_dir = self.modelpath.parent.joinpath('logs').joinpath(f"{self.model_name}-{self._lang_to_train}-finetuned")
        log_dir.mkdir(parents=True, exist_ok=True)

        # attributes to customize the training
        with wandb.init(config=config):
            config = wandb.config
            print(f"\nTraining {self.model_name} using with configurations:\n{config}")
            
            # https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/22
            args = TrainingArguments(
                save_total_limit = 2,
                output_dir = str(out_dir),
                overwrite_output_dir = True,
                logging_dir = str(log_dir),
                
                learning_rate = config.learning_rate,
                per_device_train_batch_size = config.batch_size,
                per_device_eval_batch_size = config.batch_size,
                num_train_epochs = config.epochs,
                # weight_decay = config.weight_decay,
                
                evaluation_strategy = "epoch",
                save_strategy = "epoch",  
                logging_strategy='epoch', 

                eval_accumulation_steps=1,
                metric_for_best_model = metric_name,
                push_to_hub=False, # push the model to the Hub regularly during training
                load_best_model_at_end = True,
                # report_to='wandb',  # turn on wandb logging
                )
        
        # https://huggingface.co/docs/transformers/main_classes/trainer#trainer
        trainer = Trainer(
            model_init = self.get_model,
            args = args,
            train_dataset = self._train_dataset,
            eval_dataset = self._valid_dataset,
            tokenizer = self._tokenizer,
            compute_metrics = compute_metrics,
            callbacks = [EarlyStoppingCallback(3, 0.0)]
            )
        
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        
        trainer.train()
        trainer.evaluate()
        
        print(f"free space by deleting: {out_dir}")
        shutil.rmtree(out_dir, ignore_errors=True)

# mytrainer = trainer.train_eval()