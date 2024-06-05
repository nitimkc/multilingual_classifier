## 4-*- coding: utf-8 -*-
"""
1. Evaluate the model performance per language.
2. Evaluate performance per category during each epoch.

"""
import re
import json
import numpy as np
import pandas as pd

import evaluate
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

def evaluation_display(y, y_pred, savepath, labels_map=None, model_name=None, plot=False):
    
    if labels_map is not None:
        labels_idx = [k for k,v in labels_map.items() if k in y.unique()]
        labels_name = [v for k,v in labels_map.items() if k in y.unique()]
    else:
        labels_idx = None
        labels_name = None
        
    f1_macro = f1_score(y, y_pred, labels=labels_idx, average='macro')
    f1_weighted = f1_score(y, y_pred, labels=labels_idx, average='weighted')
    acc = accuracy_score(y, y_pred)
    print(f"f1 macro: {f1_macro}\nf1 weighted: {f1_weighted}\nacc: {acc}") 

    # y_score = pred probabilityes
    # fpr, tpr, _ = roc_curve(y_test, y_score)
    # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        
    class_report = classification_report(y, y_pred, labels=labels_idx, target_names=labels_name, output_dict=True)
    # class_report = pd.DataFrame(class_report).transpose().round(2)
    
    cm = confusion_matrix(y, y_pred, normalize='true')
    cm = pd.DataFrame(cm).to_dict()
    print(f"classification report: {class_report}\nconfusion matrix: {cm}") 
    # cm = confusion_matrix(y, y_pred, labels=labels_idx, normalize='true')
    # cm = multilabel_confusion_matrix(y, y_pred)
    
    if plot:
        fig, ax = plt.subplots(figsize=(4,7))
        cm_disp = ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues', colorbar=False)
        c_bar = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.05, ax.get_position().height])
        plt.colorbar(cm_disp.im_,  cax=c_bar)
        plt.show()

    result_dict = {'model_name': model_name,
                   'f1_macro':f1_macro,
                   'f1_weighted': f1_weighted,
                   'accuracy': acc,
                   'class_report': class_report,
                   'confusion_matrix': cm}
    if savepath:
        with open(savepath.joinpath(f"evaluations.jsonl"), 'a', encoding="utf-8") as file:
            json.dump(result_dict, file, default=str)
            file.write('\n') 

class PredictionEvaluater(object):

    def __init__(self, prediction_set, target_names=None, savepath=None, model_name=None):     

        self._labels, self._predictions = prediction_set
        
        self._target_names = target_names
        self.num_labels = len(target_names)
        self.label_map = {k:v for k,v in zip(range(self.num_labels), self._target_names)}
        print(f"Number of labels in target_names is {self.num_labels}")

        self.savepath = savepath
        self.savepath.mkdir(parents=True, exist_ok=True) 
        
        self._model_name = model_name if model_name is not None else "model"
        
    def compute_metrics(self, eval_pred, eval_metric="accuracy"):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metric = evaluate.load(eval_metric)
        return metric.compute(predictions=preds, references=labels)
    
    def append_predictions(self, test_df):
        # append prediction and encoded labels to original test data
        df = test_df.copy()
        df[f"{self._model_name}_prediction"] = self._predictions
        df["annotation"] = self._labels
        if self.savepath is not None:
            sub_df = df[[f"{self._model_name}_prediction","annotation"]]
            sub_df.rename_axis('index').to_csv(self.savepath.joinpath(f"{self._model_name}_predictions.csv"))
        return df
        
    def evaluation_report(self, test_df, lang_eval=True, col_to_eval="lang"):
        df = self.append_predictions(test_df)
        # evaluation_display(y=df["annotation"], y_pred=df[f"{self._model_name}_prediction"], 
        #                    savepath=self.savepath, labels_map=self.label_map, 
        #                    model_name=f'{self._model_name}_evalall', plot=False)
        
        for dtype, dtype_df in df.groupby('original'):
            print(f"\nEvaluation for {'original' if dtype==True else 'translated'}")
            evaluation_display(y=dtype_df["annotation"], y_pred=dtype_df[f"{self._model_name}_prediction"],
                               savepath=self.savepath, labels_map=self.label_map,
                               model_name=f'{self._model_name}_eval{dtype}', plot=False)
            # if language evaluation is required
            if lang_eval:
                for lang, lang_df in dtype_df.groupby(col_to_eval):
                    print(f"\nEvaluation for language: {lang}")
                    evaluation_display(y=lang_df["annotation"], y_pred=lang_df[f"{self._model_name}_prediction"],
                                       savepath=self.savepath, labels_map=self.label_map,
                                       model_name=f'{self._model_name}_eval{lang}', plot=False)