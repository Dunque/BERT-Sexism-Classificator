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


# Mount Google Drive
from google.colab import drive # import drive from google colab

ROOT = "/content/drive"     # default location for the drive
print(ROOT)                 # print content of ROOT (Optional)

drive.mount(ROOT)           # we mount the google drive at /content/drive


class BERTBaseUncased(nn.Module):
    def __init__(self, bert_path, output_bert='pooler', NumberOfClasses=2):
        super(BERTBaseUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.output_bert = output_bert
        self.NumberOfClasses = NumberOfClasses
        self.OutPutHidden = nn.Linear(768 * 2, NumberOfClasses)
        self.OutPoller = nn.Linear(768, NumberOfClasses)

    def forward(
            self,
            ids,
            mask,
            token_type_ids
    ):
        o1, o2 = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids)
          
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

        inputs = self.tokenizer.encode_plus(
            comment,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length,
        )
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
    self.average_metrics =  'macro' if self.NumberOfLabels > 2 else 'binary'
    self.PathSaveFiles = PathSaveFiles
    self.MaxLen = MaxLen
    self.SaveModel = SaveModel


  def _run(self):
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
PathDataSet = 'data/'
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
LabelColumn = "LabelTask2"      ## "LabelTask1", "LabelTask2"
DataColumn = "text"          ## "text", "English" and "Spanish"
NewColumnsNames = {DataColumn:"Data",LabelColumn:"Label"}
df_train = df_train.rename(columns=NewColumnsNames)
# df_train = df_train.sample(frac=1).reset_index(drop=True)

#### Vizualise Data
df_train


######################################################
############## Moddify CODE ##########################
######################################################

## Select Data for train
LanguageTrain = 'whole'        ## 'Whole', 'en', 'es'

df_train_es = df_train.loc[df_train.loc[df_train['language']== 'es' ].index[0]:df_train.loc[df_train['language']== 'es'].index[-1]]
df_train_en = df_train.loc[df_train.loc[df_train['language']== 'en' ].index[0]:df_train.loc[df_train['language']== 'en'].index[-1]]

## Get a Stratified sample of 20% of data/rows for Test (whole/es/en)
df_test_es = df_train_es.groupby(['Label']).apply(lambda x: x.sample(frac=0.2, random_state=48))
df_test_en = df_train_en.groupby(['Label']).apply(lambda x: x.sample(frac=0.2, random_state=48))
df_test_whole = pd.concat([df_test_es,df_test_en])

#Selectin the data for the Standar Train and Test
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

  #Train
  df_train = df_train[df_train['Label'] != 0]
  df_train['Label'] = df_train['Label'].apply(lambda x : x -1)


  df_train_es = df_train_es[df_train_es['Label'] != 0]
  df_train_es['Label'] = df_train_es['Label'].apply(lambda x : x -1)

  df_train_en = df_train_en[df_train_en['Label'] != 0]
  df_train_en['Label'] = df_train_en['Label'].apply(lambda x : x -1)

  #Test
  df_test = df_test[df_test['Label'] != 0]
  df_test['Label'] = df_test['Label'].apply(lambda x : x -1)

  df_test_whole = df_test_whole[df_test_whole['Label'] != 0]
  df_test_whole['Label'] = df_test_whole['Label'].apply(lambda x : x -1)

  df_test_en = df_test_en[df_test_en['Label'] != 0]
  df_test_en['Label'] = df_test_en['Label'].apply(lambda x : x -1)

  df_test_es = df_test_es[df_test_es['Label'] != 0]
  df_test_es['Label'] = df_test_es['Label'].apply(lambda x : x -1)

#### Reset index
df_train = df_train.reset_index(drop=True)
df_train_es = df_train_es.reset_index(drop=True)
df_train_en = df_train_en.reset_index(drop=True)

df_test = df_test.reset_index(drop=True)
df_test_whole = df_test_whole.reset_index(drop=True)
df_test_en = df_test_en.reset_index(drop=True)
df_test_es = df_test_es.reset_index(drop=True)
 
df_test.head()



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
BertVersion = {'MultilingualBert':'../content/bert-base-multilingual-uncased/'}
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
Path = 'drive/MyDrive/Code/EXITS/Machine-Learning-Tweets-Classification/Bert/Results/' 
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

parameters: Bertmodel: MultilingualBert, Output: hidden, lr: 2e-05, Batch: 32, Totsl Num. Epochs: 8, Fold: 1

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


## Creating LateX Table
LabelTaskTable = FileResults
print(DfResultsTask.to_latex(multicolumn=True, multirow=False, label=LabelTaskTable))



## 10 Best resuts
MetricForBestResults = 'f1_macro' if df_train['Label'].nunique() > 2 else 'accuracy'
DfResultsTask.nlargest(n=10, columns= MetricForBestResults )


## Get best parameters from cross-validation DataFrame 
BestResultParameters = DfResultsTask.sort_values(MetricForBestResults, ascending=False)[:1].index
print(f'Best parameters : {BestResultParameters}')


## Add best parameters to variables in the final train
BertPath = BertVersion[BestResultParameters[0][0]]
BertVersion = {BestResultParameters[0][0] : BertVersion[BestResultParameters[0][0]]}
OutputBert = [BestResultParameters[0][1]]
LearningRate = [float(BestResultParameters[0][2])]
BatchSize = [int(BestResultParameters[0][3])]
Epochs = int(BestResultParameters[0][4])

## Criate dictinaril results
ResultsTaskBestParameters = { bert:{ output:{ lr:{ bat:{ epoc:{ metric:[] for metric in Metrics + ['loss']} for epoc in range(1, Epochs+1) } for bat in BatchSize} for lr in LearningRate} for output in OutputBert } for bert in BertVersion.keys() }

## Create file to save results BEST Parameters
#### Create file name
FileResultsBestModel = FileResults + 'BestModel'
#### Save the file fro results BEST Parameters
with open(Path + FileResultsBestModel + ".pkl",'wb') as f:
  pickle.dump(ResultsTaskBestParameters, f)

## Train with Best parameters

## Best parameters
BertV = BestResultParameters[0][0]
BertPath = BertVersion[BestResultParameters[0][0]]
OutputB = OutputBert[0]
lr = LearningRate[0]
Batch = BatchSize[0]
Epochs = Epochs

### Loading Bert trained weights
mx = BERTBaseUncased(bert_path=BertPath, output_bert=OutputB, NumberOfClasses=df_train['Label'].nunique())

## Split train and test
X_train = df_train['Data']
y_train = df_train['Label']
_, X_test, _, y_test = train_test_split(df_train['Data'], df_train['Label'], test_size=0.33, random_state=42)

print(f'parameters: Bertmodel: {BertV}, Output: {OutputB}, lr: {lr}, Batch: {Batch}, Totsl Num. Epochs: {Epochs}')
MoDeL = TrainModel(PathSaveFiles = Path,
                  BertVersion=BertV,
                  BertPath=BertPath,
                  OutputBert=OutputB,
                  LearningRate=lr,
                  BatchSize=Batch,
                  Epochs=Epochs,
                  FileName= FileResultsBestModel,
                  X_train=X_train, 
                  X_valid=X_test,
                  y_train=y_train,
                  y_valid=y_test,
                  SaveModel=True)


def _mp_fn(rank, flags):
  torch.set_default_tensor_type('torch.FloatTensor')
  a = MoDeL._run()

FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')


## Average and Save Results
AverageResultsTaskBestModel = AveragResults(FileName=FileResultsBestModel, Path=Path)

## Create a Data Frame
DfResultsTaskBestModel = create_Data_Frame(all_resultas=AverageResultsTaskBestModel)

### save results to a CSV file
DfResultsTaskBestModel.to_csv(Path + 'Average' + FileResultsBestModel + '_CSV_' + '.csv')

### See the Avarage results in the Pandas data Frame
DfResultsTaskBestModel


#TEST

class BERTDatasetTest:
    def __init__(self, comment_text, tokenizer, max_length):
        self.comment_text = comment_text
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, item):
        comment_text = str(self.comment_text[item])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length,
        )
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
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }

## Bert tozenizer
tokenizer = transformers.BertTokenizer.from_pretrained(BertPath, do_lower_case=True)

## Loading the best model
device = torch.device("xla")
model = BERTBaseUncased(bert_path=BertPath, output_bert=OutputB, NumberOfClasses=df_train['Label'].nunique()).to(device)
FileBestModel = Path + FileResultsBestModel + '.bin'
model.load_state_dict(torch.load(FileBestModel))
model.eval()


#TEST WHOLE DATA


## Prepresing the data
valid_dataset = BERTDatasetTest(
        comment_text=df_test_whole['Data'].values,
        tokenizer=tokenizer,
        max_length=110
)

valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=Batch,
    drop_last=False,
    num_workers=4,
    shuffle=False
)

## Making the Inferences
with torch.no_grad():
    fin_outputs = []
    for bi, d in tqdm(enumerate(valid_data_loader)):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        outputs_np = outputs.detach().cpu().numpy().tolist()
        fin_outputs.extend(outputs_np)



## List with Results
fin_outputs

## create a Dataframe from List of Results
df_results = pd.DataFrame.from_records(fin_outputs)

## get the model inference
df_results['Inference'] = df_results.idxmax(axis=1)

## Visualize results
df_results.head()



## Get rows index
df_idex = df_test_whole.loc[:,["id", "Label"]]

## Add index to the Results dataframe
df_results = df_results.join(df_idex)

### save results to a CSV file
df_save_results = df_results.copy()
if df_train['Label'].nunique() > 2:
  df_save_results = df_save_results.rename({0:1, 1:2, 2:3, 3:4, 4:5}, axis='columns')
  df_save_results['Label'] = df_save_results['Label'].apply( lambda x : x+1)
  df_save_results['Inference'] = df_save_results['Inference'].apply(lambda x : x+1)
  
df_save_results.to_csv(Path + 'ModelInfereneces_' + FileResultsBestModel + '_WholeSetTest' +'_CSV_' + '.csv')

## ## Visualize results
df_results.head()


## caculation of performace metric
Target = df_results[df_results.columns[-1]].tolist()
Output = df_results[df_results.columns[-3]].tolist()

average_metrics = 'macro' if df_train['Label'].nunique() > 2 else 'binary'
print(f'Accuracy : {metrics.accuracy_score(Target, Output)}')
print(f'Recall : {metrics.recall_score(Target, Output, average = average_metrics)}')
print(f'Precision : {metrics.precision_score(Target, Output, average = average_metrics)}')
print(f'f1-score : {metrics.f1_score(Target, Output, average= average_metrics)}')


## caculation of performace metric
Target = df_results[df_results.columns[-1]].tolist()
Output = df_results[df_results.columns[-3]].tolist()

average_metrics = 'macro' if df_train['Label'].nunique() > 2 else 'binary'
print(f'Accuracy : {metrics.accuracy_score(Target, Output)}')
print(f'Recall : {metrics.recall_score(Target, Output, average = average_metrics)}')
print(f'Precision : {metrics.precision_score(Target, Output, average = average_metrics)}')
print(f'f1-score : {metrics.f1_score(Target, Output, average= average_metrics)}')


#INFERENCE

# Load data for inference 

#### Data Path
PathDataSet = 'data/'
FileDataset = 'EXIST2021_translatedTest'
#### Load tsv as a Data Frame
df_RealData = pd.read_csv(PathDataSet + FileDataset + '.csv', index_col=0)

#### Change columns names for the train
df_RealData = df_RealData.rename(columns=NewColumnsNames)

#### Vizualise Data
df_RealData.head()


## Prepresing the data
valid_dataset = BERTDatasetTest(
        comment_text=df_RealData['Data'].values,
        tokenizer=tokenizer,
        max_length=110
)

valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=Batch,
    drop_last=False,
    num_workers=4,
    shuffle=False
)


## Making the Inferences
with torch.no_grad():
    fin_outputs = []
    for bi, d in tqdm(enumerate(valid_data_loader)):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        outputs_np = outputs.detach().cpu().numpy().tolist()
        fin_outputs.extend(outputs_np)


## List with Results
fin_outputs

## create a Dataframe from List of Results
df_results = pd.DataFrame.from_records(fin_outputs)

## change columns if task2
if df_train['Label'].nunique() > 2:
  df_results = df_results.rename({0:1, 1:2, 2:3, 3:4, 4:5}, axis='columns')

## get the model inference
df_results['Inference'] = df_results.idxmax(axis=1)

## Visualize results
df_results.head()


## Get rows index
df_idex = df_RealData.loc[:,["id"]]

## Add index to the Results dataframe
df_results = df_results.join(df_idex)

### save results to a CSV file
df_results.to_csv(Path + 'ModelInfereneces' + FileResultsBestModel + '_RealData' + '_CSV_' + '.csv')

## ## Visualize results
df_results.head()



#MULTICLASS


# Path = 'drive/MyDrive/Code/EXITS/Machine-Learning-Tweets-Classification/Bert/Results/EnglishBert_enDataTrain/'
# File = 'EnglishBertTask2enDataTrain_Results'



# import pickle
# with open('drive/MyDrive/Code/EXITS/Machine-Learning-Tweets-Classification/Bert/Results/EnglishBert_enDataTrain/EnglishBertTask2enDataTrain_Results' + ".pkl", "rb") as f:
#   Re = pickle.load(f)
# Re

# DfResultsTask = pd.read_csv(Path + 'AverageSpanishBertTask1Results_CSV_.csv', index_col=[0,1], skipinitialspace=True)

# def CleanBrokeTrain(FileName, Path, NumberOfFoldes=10):
#   with open(Path + FileName + ".pkl", "rb") as f:
#               Results = pickle.load(f)

#   for BT, ModelBertType,  in Results.items():
#     for OP, OutPut in ModelBertType.items():
#       for LR, LearningRate in OutPut.items():
#         for BS, BatchSize in LearningRate.items():
#           for EP, Epoch in BatchSize.items():
#             for Metrics, ValuesCrossValidation in  Epoch.items():
 
#               if len(ValuesCrossValidation) != 0 and not len(ValuesCrossValidation) == NumberOfFoldes:
#                 Results[BT][OP][LR][BS][EP][Metrics] = []
            
#   with open(FileName + '.pkl','wb') as f:
#     pickle.dump(Results, f)

#   with open(Path + FileName + '.pkl','wb') as f:
#     pickle.dump(Results, f)

# CleanBrokeTrain(FileName=FileResults, Path=Path, NumberOfFoldes=10)

# LengPhrase = df_train['text'].str.split().str.len().tolist()
# LengPhrase.sort()
# LengPhrase[-13:]

# LengPhrase = df_RealData['text'].str.split().str.len().tolist()
# LengPhrase.sort()

# LengPhrase[-50:]


#ALTERNATIVE CODE INFERENCE

# class BERTDatasetTest:
#     def __init__(self, comment_text, targets, tokenizer, max_length):
#         self.comment_text = comment_text
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.targets = targets

#     def __len__(self):
#         return len(self.comment_text)

#     def __getitem__(self, item):
#         comment_text = str(self.comment_text[item])
#         comment_text = " ".join(comment_text.split())

#         inputs = self.tokenizer.encode_plus(
#             comment_text,
#             None,
#             truncation=True,
#             add_special_tokens=True,
#             max_length=self.max_length,
#         )
#         ids = inputs["input_ids"]
#         token_type_ids = inputs["token_type_ids"]
#         mask = inputs["attention_mask"]
        
#         padding_length = self.max_length - len(ids)
        
#         ids = ids + ([0] * padding_length)
#         mask = mask + ([0] * padding_length)
#         token_type_ids = token_type_ids + ([0] * padding_length)
        
#         return {
#             'ids': torch.tensor(ids, dtype=torch.long),
#             'mask': torch.tensor(mask, dtype=torch.long),
#             'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
#             'targets': torch.tensor(self.targets[item], dtype=torch.float)
#         }

# ## Bert tozenizer
# tokenizer = transformers.BertTokenizer.from_pretrained(BertPath, do_lower_case=True)

# ## Loading the best model
# device = torch.device("xla")
# model = BERTBaseUncased(bert_path=BertPath, output_bert=OutputB, NumberOfClasses=df_train['Label'].nunique()).to(device)
# FileBestModel = Path + FileResultsBestModel + '.bin'
# model.load_state_dict(torch.load(FileBestModel))
# model.eval()


#TEST WHOLE DATA

# ## Prepresing the data
# valid_dataset = BERTDatasetTest(
#         comment_text=df_test_whole['Data'].values,
#         targets=df_test_whole['Label'].values,
#         tokenizer=tokenizer,
#         max_length=110
# )

# valid_data_loader = torch.utils.data.DataLoader(
#     valid_dataset,
#     batch_size=Batch,
#     drop_last=False,
#     num_workers=4,
#     shuffle=False
# )

# with torch.no_grad():
#           model.eval()
#           fin_targets = []
#           fin_outputs = []
#           for bi, d in tqdm(enumerate(valid_data_loader)):
#               ids = d["ids"]
#               mask = d["mask"]
#               token_type_ids = d["token_type_ids"]
#               targets = d["targets"]

#               ids = ids.to(device, dtype=torch.long)
#               mask = mask.to(device, dtype=torch.long)
#               token_type_ids = token_type_ids.to(device, dtype=torch.long)
#               targets = targets.to(device, dtype=torch.float)

#               outputs = model(
#                   ids=ids,
#                   mask=mask,
#                   token_type_ids=token_type_ids
#               )

#               targets_np = targets.cpu().detach().numpy().tolist()
#               outputs = torch.argmax(outputs, dim=1)
#               outputs_np = outputs.detach().cpu().numpy().tolist()

#               fin_targets.extend(targets_np)
#               fin_outputs.extend(outputs_np)

# ## caculation of performace metric
# Target = fin_targets
# Output = fin_outputs

# average_metrics = 'macro' if df_train['Label'].nunique() > 2 else 'binary'
# print(f'Accuracy : {metrics.accuracy_score(Target, Output)}')
# print(f'Recall : {metrics.recall_score(Target, Output, average = average_metrics)}')
# print(f'Precision : {metrics.precision_score(Target, Output, average = average_metrics)}')
# print(f'f1-score : {metrics.f1_score(Target, Output, average= average_metrics)}')




# import collections

# t=collections.Counter(Target)
# print(t)
# o=collections.Counter(Output)
# print(o)

