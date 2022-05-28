# !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# !python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev

# !apt-get install git-lfs

# !git lfs install
# !git clone https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased

# !git lfs install
# !git clone https://huggingface.co/bert-base-uncased

# !git lfs install
# !git clone https://huggingface.co/bert-base-multilingual-uncased

# !pip install transformers==3

### add NLP dependences
import pickle
import os
import torch
import pandas as pd
from scipy import stats
import numpy as np
import os

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split

from tqdm import tqdm
from collections import OrderedDict, namedtuple
import torch.nn as nn
from torch.optim import lr_scheduler
import joblib

import logging
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
import sys
from sklearn import metrics, model_selection

import warnings
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import warnings

from torch_xla.core.xla_model import mesh_reduce

warnings.filterwarnings("ignore")

#base neural network class
class BERTBaseUncased(nn.Module):
    def __init__(self, bert_path, output_bert='pooler', NumberOfClasses=2):
        super(BERTBaseUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        #effective technique for regularization and preventing the co-adaptation of neurons 
        self.bert_drop = nn.Dropout(0.3)
        self.output_bert = output_bert
        self.NumberOfClasses = NumberOfClasses
        self.OutPutHidden = nn.Linear(768 * 2, NumberOfClasses)
        self.OutPoller = nn.Linear(768, NumberOfClasses)

    def forward(self, ids, mask, token_type_ids):
        o1, o2 = self.bert(ids,attention_mask=mask,token_type_ids=token_type_ids)
          
        if self.output_bert=='hidden':
          apool = torch.mean(o1, 1)
          mpool, _ = torch.max(o1, 1)
          cat = torch.cat((apool, mpool), 1)
          bo = self.bert_drop(cat)

          output = self.OutPutHidden(bo) 

        else:
          bo = self.bert_drop(o2)
          output = self.OutPoller(bo)
        
        return output

class BERTDatasetTraining:
    def __init__(self, comment, targets, tokenizer, max_length):
        self.comment = comment
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.targets = targets

    def __len__(self):
        return len(self.comment)

    def __getitem__(self, item):
        comment = str(self.comment[item])
        comment = " ".join(comment.split())

        #Tokenizes, adds [CLS] and [SEP], pads and truncates to max length, creates attention masks.
        inputs = self.tokenizer.encode_plus(comment,None,truncation=True,add_special_tokens=True,max_length=self.max_length)
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = self.max_length - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float)
        }

class TrainModel():
  def __init__(self, PathSaveFiles, BertVersion, BertPath,  OutputBert, LearningRate, BatchSize, Epochs, FileName, X_train, X_valid, y_train ,y_valid, MaxLen = 110, SaveModel=False):
    self.BertVersion = BertVersion
    self.BertPath = BertPath
    self.OutputBert = OutputBert
    self.LearningRate = LearningRate
    self.BatchSize = BatchSize
    self.Epochs = Epochs
    self.FileName = FileName
    self.X_train = X_train
    self.X_valid = X_valid
    self.y_train = y_train
    self.y_valid = y_valid
    self.NumberOfLabels = y_train.nunique()
    #Task 2 or task 1
    self.average_metrics =  'macro' if self.NumberOfLabels > 2 else 'binary'
    self.PathSaveFiles = PathSaveFiles
    self.MaxLen = MaxLen
    self.SaveModel = SaveModel


  def _run(self):

      #Seriaization using pickle
      def OpenEndSave(CurrentEpoch, module):
          if module == 'open'and CurrentEpoch == 1:
            with open(self.PathSaveFiles + self.FileName + ".pkl", "rb") as f:
              self.Results = pickle.load(f)

          elif module == 'save' and CurrentEpoch == self.Epochs:
            with open(self.PathSaveFiles + self.FileName + ".pkl",'wb') as f:
              pickle.dump(self.Results, f)


      def loss_fn(outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)
            

      def train_loop_fn(data_loader, model, optimizer, device, scheduler=None, epoch=None):
          model.train()
          for bi, d in enumerate(data_loader):
              ids = d["ids"]
              mask = d["mask"]
              token_type_ids = d["token_type_ids"]
              targets = d["targets"]

              ids = ids.to(device, dtype=torch.long)
              mask = mask.to(device, dtype=torch.long)
              token_type_ids = token_type_ids.to(device, dtype=torch.long)
              targets = targets.to(device, dtype=torch.float)
              

              optimizer.zero_grad()
              outputs = model(
                  ids=ids,
                  mask=mask,
                  token_type_ids=token_type_ids
              )

              loss = loss_fn(outputs, targets)
              if bi % 10 == 0:
                  xm.master_print(f'bi={bi}, loss={loss}')

                  ValueLoss = loss.cpu().detach().numpy().tolist()
                  ValueLoss = xm.mesh_reduce('test_loss',ValueLoss, np.mean)
                  self.Results[self.BertVersion][self.OutputBert][self.LearningRate][self.BatchSize][epoch]['loss'].append(ValueLoss)

              loss.backward()
              xm.optimizer_step(optimizer)
              if scheduler is not None:
                  scheduler.step()

      #This eval loop is designet to work with distributed data paralell in pytorch
      def eval_loop_fn(data_loader, model, device):
          model.eval()
          fin_targets = []
          fin_outputs = []
          for bi, d in enumerate(data_loader):
              ids = d["ids"]
              mask = d["mask"]
              token_type_ids = d["token_type_ids"]
              targets = d["targets"]

              ids = ids.to(device, dtype=torch.long)
              mask = mask.to(device, dtype=torch.long)
              token_type_ids = token_type_ids.to(device, dtype=torch.long)
              targets = targets.to(device, dtype=torch.float)

              outputs = model(
                  ids=ids,
                  mask=mask,
                  token_type_ids=token_type_ids
              )

              targets_np = targets.cpu().detach().numpy().tolist()
              outputs = torch.argmax(outputs, dim=1)
              outputs_np = outputs.detach().cpu().numpy().tolist()

              fin_targets.extend(targets_np)
              fin_outputs.extend(outputs_np)    

          return fin_outputs, fin_targets

      # tokenizer
      tokenizer = transformers.BertTokenizer.from_pretrained(self.BertPath, do_lower_case=True)

      train_dataset = BERTDatasetTraining(
          comment=self.X_train.values,
          targets=self.y_train.values,
          tokenizer=tokenizer,
          max_length=self.MaxLen
      )

      train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)

      train_data_loader = torch.utils.data.DataLoader(
          train_dataset,
          batch_size=self.BatchSize,
          sampler=train_sampler,
          drop_last=True,
          num_workers=1
      )

      valid_dataset = BERTDatasetTraining(
          comment=self.X_valid.values,
          targets=self.y_valid.values,
          tokenizer=tokenizer,
          max_length=self.MaxLen
      )

      valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False)

      valid_data_loader = torch.utils.data.DataLoader(
          valid_dataset,
          batch_size=16,
          sampler=valid_sampler,
          drop_last=False,
          num_workers=1
      )

      device = xm.xla_device()
      model = mx.to(device)
      

      param_optimizer = list(model.named_parameters())
      no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
      optimizer_grouped_parameters = [
          {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
          {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

      
      lr = 0.4 * self.LearningRate * xm.xrt_world_size()
      num_train_steps = int(len(train_dataset) / self.BatchSize / xm.xrt_world_size() * self.Epochs)
      xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')

      optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
      scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=0,
          num_training_steps=num_train_steps
      )

      best_f1, f1, best_cem, cem = 0,0,0,0

      for epoch in range(1, self.Epochs+1):
        ## print epoch
          xm.master_print(f'Epoch: {epoch} of {self.Epochs}')
        ## Open file to save results
          OpenEndSave(CurrentEpoch=epoch, module='open')

          para_loader = pl.ParallelLoader(train_data_loader, [device])
          train_loop_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler, epoch=epoch)

          para_loader = pl.ParallelLoader(valid_data_loader, [device])
          o, t = eval_loop_fn(para_loader.per_device_loader(device), model, device)
          
          if self.NumberOfLabels == 2:
            f1 = xm.mesh_reduce('validation_f1', metrics.f1_score(t, o), np.mean)
            self.Results[self.BertVersion][self.OutputBert][self.LearningRate][self.BatchSize][epoch]['f1'].append(f1)

          else:
            self.Results[self.BertVersion][self.OutputBert][self.LearningRate][self.BatchSize][epoch]['f1_macro'].append(xm.mesh_reduce('validation_f1_macro', metrics.f1_score(t, o, average=self.average_metrics), np.mean))
            self.Results[self.BertVersion][self.OutputBert][self.LearningRate][self.BatchSize][epoch]['f1_weighted'].append(xm.mesh_reduce('validation_f1_weighted', metrics.f1_score(t, o, average='weighted'), np.mean))
            # cem = xm.mesh_reduce('validation_cem', cem_metric(t, o), np.mean)
            # self.Results[self.BertVersion][self.OutputBert][self.LearningRate][self.BatchSize][epoch]['cem'].append(xm.mesh_reduce('validation_cem', cem_metric(t, o), np.mean))

          accuracy = metrics.accuracy_score(t, o)
          accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
          self.Results[self.BertVersion][self.OutputBert][self.LearningRate][self.BatchSize][epoch]['accuracy'].append(accuracy)
          self.Results[self.BertVersion][self.OutputBert][self.LearningRate][self.BatchSize][epoch]['recall'].append(xm.mesh_reduce('validation_recall', metrics.recall_score(t, o, average=self.average_metrics), np.mean))
          self.Results[self.BertVersion][self.OutputBert][self.LearningRate][self.BatchSize][epoch]['precision'].append(xm.mesh_reduce('validation_precison', metrics.precision_score(t, o, average=self.average_metrics), np.mean))
              
        ## save file with save results
          OpenEndSave(CurrentEpoch=epoch, module='save')

        ## Save model
          if self.SaveModel and epoch == self.Epochs:
            xm.save(model.state_dict(), self.PathSaveFiles + self.FileName + '.bin')
        
        ## print accuracy
          xm.master_print(f'Accuracy = {accuracy}')



# Load Data

#### Data Path
PathDataSet = '../content/drive/MyDrive/Code/EXITS/Data/'
FileDataset = 'EXIST2021_translatedTraining'
#### Load tsv as a Data Frame
df_train = pd.read_csv(PathDataSet + FileDataset + '.csv', index_col=0)

#### Create two new columns converting str labels to Num label
df_train['LabelTask1'] = df_train['task1'].apply(lambda x : 1 if x == 'sexist' else 0)
CategorisList = list(df_train.task2.unique())
CategorisList.remove('non-sexist')
CategorisList.insert(0,'non-sexist')
CategoriSexism = {CategorisList[index]: index for index in range(len(list(df_train.task2.unique())))}
df_train['LabelTask2'] = df_train['task2'].apply(lambda x : CategoriSexism[x])

#### Get columns names
TestColumnNames = list(df_train.columns)
#### Vizualise Data
df_train.head()

######################################################
############## Moddify CODE ##########################
######################################################

#### Change columns names for the train
LabelColumn = "LabelTask1"      ## "LabelTask1", "LabelTask2"
DataColumn = "English"          ## "text", "English" and "Spanish"
NewColumnsNames = {DataColumn:"Data",LabelColumn:"Label"}
df_train = df_train.rename(columns=NewColumnsNames)
# df_train = df_train.sample(frac=1).reset_index(drop=True)

#### Vizualise Data
df_train



######################################################
############## Moddify CODE ##########################
######################################################

## Select Data for train
LanguageTrain = 'en'        ## 'Whole', 'en', 'es'

df_train_es = df_train.loc[df_train.loc[df_train['language']== 'es' ].index[0]:df_train.loc[df_train['language']== 'es'].index[-1]]
df_train_en = df_train.loc[df_train.loc[df_train['language']== 'en' ].index[0]:df_train.loc[df_train['language']== 'en'].index[-1]]

## Get a Stratified sample of 20% of data/rows for Test (whole/es/en)
df_test_es = df_train_es.groupby(['Label']).apply(lambda x: x.sample(frac=0.2, random_state=48))
df_test_en = df_train_en.groupby(['Label']).apply(lambda x: x.sample(frac=0.2, random_state=48))
df_test_whole = pd.concat([df_test_es,df_test_en])

#Selectin the data for the Standard Train and Test
if LanguageTrain == 'whole':
  df_test = df_test_whole
elif LanguageTrain == 'es':
  df_test = df_test_es
  df_train = df_train_es
elif LanguageTrain == 'en':
  df_test = df_test_en
  df_train = df_train_en
else:
  print('wrong data')

df_test.head()


# Removing Extra Index levels
df_test_es = df_test_es.reset_index(level=0, drop=True)
df_test_en = df_test_en.reset_index(level=0, drop=True)
df_test_whole = df_test_whole.reset_index(level=0, drop=True)

# Importantt for remove index in the next cell
df_test = df_test.reset_index(level=0, drop=True)

# Checking the Data
df_test.head()

# Remove the data/rows used for test set from the train set
df_train = df_train.drop(df_test.index)
df_train.head()



# Reset index datframes and and Remove non-sexist rows if task 2 
#### Remove non-sexist rows if task 2 
if df_train['Label'].nunique() > 2:

  df_train = df_train[df_train['Label'] != 0]
  df_train_es = df_train_es[df_train_es['Label'] != 0]
  df_train_en = df_train_en[df_train_en['Label'] != 0]

  df_test = df_test[df_test['Label'] != 0]
  df_test_whole = df_test_whole[df_test_whole['Label'] != 0]
  df_test_en = df_test_en[df_test_en['Label'] != 0]
  df_test_es = df_test_es[df_test_es['Label'] != 0]

#### Reset index
df_train = df_train.reset_index(drop=True)
df_train_es = df_train_es.reset_index(drop=True)
df_train_en = df_train_en.reset_index(drop=True)

df_test = df_test.reset_index(drop=True)
df_test_whole = df_test_whole.reset_index(drop=True)
df_test_en = df_test_en.reset_index(drop=True)
df_test_es = df_test_es.reset_index(drop=True)
 
df_test.head()


#LOAD WEIGHTS
def CriateFileName(BertVersionDict, NumberOfClasses):
  
  NameFile = str()
  for BertModel in BertVersionDict.keys():
    NameFile += BertModel

  if NumberOfClasses > 2:
    NameFile += 'Task2'
  else:
    NameFile += 'Task1'

  return NameFile

######################################################
############## Moddify CODE - BERT model #############
######################################################

## Train Parameters
BertVersion = {'EnglishBert':'../content/bert-base-uncased/'}
OutputBert = ['hidden', 'pooler']
LearningRate = [2e-5, 3e-5, 5e-5]
BatchSize = [32, 64]
Epochs = 8

## Evalute matrics
###### Task 1
MetricsTask1 = ['accuracy', 'f1', 'recall', 'precision']
###### Task 2
MetricsTask2 = ['accuracy', 'f1_macro', 'f1_weighted', 'recall', 'precision']

## Get for 'Binary' classification' task1 or 'Multilabel classifcation' task2
Metrics = MetricsTask2 if df_train['Label'].nunique() > 2 else MetricsTask1

## Criate dictinaril results
ResultsTask = { bert:{ output:{ lr:{ bat:{ epoc:{ metric:[] for metric in Metrics + ['loss']} for epoc in range(1, Epochs+1) } for bat in BatchSize} for lr in LearningRate} for output in OutputBert } for bert in BertVersion.keys() }

## Where to Save Files
Path = 'results/' 
BertModels = ''
for b in list(BertVersion.keys()):
  BertModels =  BertModels  + b + '_'
Folder = BertModels + LanguageTrain
Path = Path + Folder + 'DataTrain' + '/'

## Criate file to save results if it does not exist 
if not os.path.exists(Path):
  print(f'Criate folder : {Folder}' )
  print(f'Path : {Path}')
  os.makedirs(Path)

## Creating Main Parte Bert File Name
MainParteBertFileName = CriateFileName(BertVersion, NumberOfClasses=df_train['Label'].nunique()) + LanguageTrain

## Create file to save results if it does not existe
FileResults = MainParteBertFileName + 'DataTrain' + '_Results'
if not os.path.exists(Path + FileResults + '.pkl'):
  print(f'Creating File for results : {FileResults}.pkl')
  print(f'File Path : {Path}')
  with open(Path + FileResults + ".pkl",'wb') as f:
    pickle.dump(ResultsTask, f)



#TRAIN

### Cross Validation
for BertV, BertPath in BertVersion.items():
  for OutputB in OutputBert:

    ### Loading Bert trained weights
    mx = BERTBaseUncased(bert_path=BertPath, output_bert=OutputB, NumberOfClasses=df_train['Label'].nunique())

    for lr in LearningRate:
      for Batch in BatchSize:

        ## StratifiedKFold
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=48)
        fold = 1
        for train_index, valid_index in skf.split(df_train['Data'], df_train['Label']):
          X_train, X_valid = df_train.loc[train_index, 'Data'], df_train.loc[valid_index, 'Data']
          y_train, y_valid = df_train.loc[train_index, 'Label'], df_train.loc[valid_index, 'Label']

          print(f'parameters: Bertmodel: {BertV}, Output: {OutputB}, lr: {lr}, Batch: {Batch}, Totsl Num. Epochs: {Epochs}, Fold: {fold}')
          fold += 1
          MoDeL = TrainModel(PathSaveFiles = Path,
                            BertVersion=BertV,
                            BertPath=BertPath,
                            OutputBert=OutputB,
                            LearningRate=lr,
                            BatchSize=Batch,
                            Epochs=Epochs,
                            FileName= FileResults,
                            X_train=X_train, 
                            X_valid=X_valid,
                            y_train=y_train,
                            y_valid=y_valid)
        

          def _mp_fn(rank, flags):
            torch.set_default_tensor_type('torch.FloatTensor')
            a = MoDeL._run()

          FLAGS={}
          xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')



#RESULTS

def AveragResults(FileName, Path):
  with open(Path + FileName + ".pkl", "rb") as f:
              Results = pickle.load(f)

  for BT, ModelBertType,  in Results.items():
    for OP, OutPut in ModelBertType.items():
      for LR, LearningRate in OutPut.items():
        for BS, BatchSize in LearningRate.items():
          for EP, Epoch in BatchSize.items():
            for Metrics, ValuesCrossValidation in  Epoch.items():
 
              # Metrics = np.mean(ValuesCrossValidation)
              Results[BT][OP][LR][BS][EP][Metrics] = np.mean(ValuesCrossValidation)
            
  with open('Average' + FileName + '.pkl','wb') as f:
    pickle.dump(Results, f)

  with open(Path + 'Average' + FileName + '.pkl','wb') as f:
    pickle.dump(Results, f)
  
  return Results

## Average and Save Results
AverageResultsTask = AveragResults(FileName=FileResults, Path=Path)

### create dataframe for our results
def create_Data_Frame(all_resultas):

  

  ### Criate a pandas da Frame with all results
  df_results = pd.DataFrame.from_dict({(BertType, OutpuType, LearningRate, BactSize, Epochs): all_resultas[BertType][OutpuType][LearningRate][BactSize][Epochs]
                            for BertType in all_resultas.keys()
                            for OutpuType in all_resultas[BertType].keys()
                            for LearningRate in all_resultas[BertType][OutpuType].keys()
                            for BactSize in all_resultas[BertType][OutpuType][LearningRate].keys()
                            for Epochs in all_resultas[BertType][OutpuType][LearningRate][BactSize].keys()},
                        orient='index')
  return df_results

## Create a Data Frame
DfResultsTask = create_Data_Frame(all_resultas=AverageResultsTask)

### save results to a CSV file
DfResultsTask.to_csv(Path + 'Average' + FileResults + '_CSV_' + '.csv')

### See the Avarage results in the Pandas data Frame
DfResultsTask

