import os
import random
import re
import time
# Main support libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# TRAINING
import torch
import torch.nn as nn
# Whole dataset training evaluation (softmax)
import torch.nn.functional as F
# Evaluation on validation set
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
# Spitting the date into train and validation
from sklearn.model_selection import train_test_split
# DATALOADER
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel
# TOKENIZER
from transformers import BertTokenizer
# OPTIMIZER
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
# Ray Fine tuner
from ray import tune
from ray.tune import CLIReporter


absolutePath = "/home/roi_santos_rios/Desktop/BERT-Sexism-Classificator/"
modelPath = absolutePath + "models/multiEngBertSynonyms/"
translated_data_path = absolutePath + "data/EXIST2021_translatedTrainingAugmented.csv"
translated_test_data_path = absolutePath + "data/EXIST2021_translatedTest.csv"

plt.rcParams.update({'font.family': 'serif'})
plt.style.use("seaborn-whitegrid")


# Function that returns two dataframes, for training and test
# Paths must be absolute in order for the threads to work
def load_data(translated_data=translated_data_path,
              translated_test_data=translated_test_data_path):
    # Load data and set labels
    data = pd.read_csv(translated_data)

    # Plot label distribution
    plot = plt.subplot()
    data.groupby('task2').size().plot.bar()
    plot.set_xlabel('labels', fontsize=10)
    plot.set_xticklabels(["II", "MNSV", "NS", "O", "SV", "SD"])
    plt.xticks(rotation=0)
    plot.xaxis.set_label_position('bottom')
    plot.xaxis.tick_bottom()

    values = data.groupby('task2').size()

    for i in range(6):
        plt.text(i, values[i]//2, values[i], ha='center')

    plot.set_ylabel('amount of tweets', fontsize=10)
    plt.yticks(rotation=90)

    plt.tight_layout()
    plt.savefig(modelPath+"labelDistribution.png")

    # convert labels to integers
    category_list = list(data.task2.unique())
    category_list.remove('non-sexist')
    category_list.insert(0, 'non-sexist')
    category_sexism = {category_list[index]: index for index in range(len(list(data.task2.unique())))}
    data['LabelTask2'] = data['task2'].apply(lambda x: category_sexism[x])

    # English version
    data = data[['id', 'English', 'LabelTask2']]
    data = data.rename(columns={'id': 'id', 'English': 'tweet', 'LabelTask2': 'label'})

    # Display 5 random samples
    data.sample(5)

    # Load test data
    test_data = pd.read_csv(translated_test_data)

    # Keep important columns
    # English
    test_data = test_data[['id', 'English']]
    test_data = test_data.rename(columns={'id': 'id', 'English': 'tweet'})

    # Display 5 samples from the test data
    test_data.sample(5)

    return data, test_data, category_sexism


# Small tweet text preprocessing
def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove links
    text = re.sub(r"http\S+", "", text)

    return text


def preprocessing_for_bert(data, tokenizer, max_len):
    """Perform required preprocessing steps for pretrained BERT.
    @param max_len:
    @param tokenizer:
    @param    data: (np.array) Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=max_len,  # Max length to truncate/pad
            padding='max_length',  # Pad sentence to max length
            truncation=True,  # Truncate to adapt to the 512 token limit
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 6

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids: (torch.Tensor) an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask: (torch.Tensor) a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits: (torch.Tensor) an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


# TRAINING LOOP
# This function is designed ot fit in the ray tune run function, it has the training and evaluation
# parts integrated, as well as the dataloader. This has to be done this way so it can be executed on
# different threads
def train_model_hyperparams(config, checkpoint_dir=None):
    """Train the BertClassifier model."""

    # Try to get the gpu to work instead of the cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Load data
    data, test_data, category_sexism = load_data()

    # Spitting the date into train_model and validation
    x = data.tweet.values
    y = data.label.values

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=2020)

    # TOKENIZE
    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # LENGTH
    max_len = 64

    # Run function `preprocessing_for_bert` on the train_model set and the validation set
    train_inputs, train_masks = preprocessing_for_bert(x_train, tokenizer, max_len)
    val_inputs, val_masks = preprocessing_for_bert(x_val, tokenizer, max_len)

    # DATALOADER
    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config["batch_size"])

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=config["batch_size"])

    # BERT INITIALIZATION
    # Instantiate Bert Classifier
    model = BertClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    model.to(device)

    # OPTIMIZER
    # Create the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=config["learning_rate"],  # Default learning rate
                      eps=config["epsilon"],  # Default epsilon value
                      betas=config["betas"]
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * config["epochs"]

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()

    # Start training loop
    print("Start training...\n")
    for epoch_i in range(config["epochs"]):
        # =======================================
        #               Training
        # =======================================

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

        # =======================================
        #               Evaluation
        # =======================================
        model.eval()

        # Tracking variables
        val_accuracy = []
        val_loss = []
        val_step = 0

        # Specify loss function
        loss_fn = nn.CrossEntropyLoss()

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)

            # Compute loss
            loss = loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

            val_step += 1

        with tune.checkpoint_dir(epoch_i) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        # Compute the average accuracy and loss over the validation set.
        mean_loss = np.mean(val_loss)
        mean_accuracy = np.mean(val_accuracy)

        tune.report(loss=(mean_loss / val_step), accuracy=mean_accuracy)


def data_loaders(config):
    # Load data
    data, test_data, category_sexism = load_data()

    # Spitting the date into train_model and validation
    x = data.tweet.values
    y = data.label.values

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=2020)

    # TOKENIZE
    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # LENGTH
    # Concatenate train_model data and test data
    all_tweets = np.concatenate([data.tweet.values, test_data.tweet.values])

    # Encode our concatenated data
    encoded_tweets = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_tweets]

    # Find the maximum length
    max_len = max([len(sent) for sent in encoded_tweets])
    print('Max length: ', max_len)
    max_len = 64

    # Print sentence 0 and its encoded token ids
    token_ids = list(preprocessing_for_bert([x[0]], tokenizer, max_len)[0].squeeze().numpy())
    print('Original: ', x[0])
    print('Token IDs: ', token_ids)

    # Run function `preprocessing_for_bert` on the train_model set and the validation set
    print('Tokenizing data...')
    train_inputs, train_masks = preprocessing_for_bert(x_train, tokenizer, max_len)
    val_inputs, val_masks = preprocessing_for_bert(x_val, tokenizer, max_len)

    # Print sentence 0 and preprocessed sentence
    print('Original: ', x[0])
    print('Processed: ', text_preprocessing(x[0]))

    # DATALOADER
    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config["batch_size"])

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=config["batch_size"])

    return train_dataloader, val_dataloader, y_val, category_sexism


def train_model(config, train_dataloader, val_dataloader, category_sexism):
    """Train the BertClassifier model."""

    # Try to get the gpu to work instead of the cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # BERT INITIALIZATION
    # Instantiate Bert Classifier
    model = BertClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    model.to(device)

    # OPTIMIZER
    # Create the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=config["learning_rate"],  # Default learning rate
                      eps=config["epsilon"],  # Default epsilon value
                      betas=config["betas"]
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * config["epochs"]

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)

    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()

    # List to keep an eye on the loss of the model
    loss_list = []

    # Start training loop
    print("Start training...\n")
    for epoch_i in range(config["epochs"]):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(
            f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        loss_list.append(avg_train_loss)

        print("-" * 70)
        # =======================================
        #               Evaluation
        # =======================================
        # After the completion of each training epoch, measure the model's performance
        # on our validation set.

        # Print performance over the entire training data
        time_elapsed = time.time() - t0_epoch

        evaluate(model, device, val_dataloader, avg_train_loss, time_elapsed, epoch_i, category_sexism)

        print("\n")

    bl = plt.subplot()

    bl.set_xlabel('batch')
    bl.set_ylabel('loss')
    bl.plot(range(1, config["epochs"]+1), loss_list)
    plt.tight_layout()
    plt.savefig(modelPath+"batchLoss.png")

    print("Training complete!")

    return model


# EVALUATION
def evaluate(model, device, val_dataloader, avg_train_loss, time_elapsed, epoch_i, category_sexism):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    val_step = 0

    y_pred = []
    y_true = []

    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

        y_pred.extend(torch.argmax(logits, 1).tolist())
        y_true.extend(b_labels.tolist())

        val_step += 1

    # Compute the average accuracy and loss over the validation set.
    mean_loss = np.mean(val_loss)
    mean_accuracy = np.mean(val_accuracy)

    print(
        f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {mean_loss:^10.6f} | {mean_accuracy:^9.2f} | {time_elapsed:^9.2f}")
    print("-" * 70)

    print('Classification Report:')
    print(classification_report(y_true, y_pred, target_names=category_sexism.keys(), digits=4))
    clas_rep_file = open((modelPath + "classReport.txt"), "w")
    clas_rep_file.write(classification_report(y_true, y_pred, target_names=category_sexism.keys(), digits=4))

    print("\n")

    cm = confusion_matrix(y_true, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=12)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=30)
    ax.xaxis.set_ticklabels(category_sexism.keys(), fontsize=8)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=12)
    ax.yaxis.set_ticklabels(category_sexism.keys(), fontsize=8)
    plt.yticks(rotation=30)

    plt.title('Refined Confusion Matrix', fontsize=15)
    plt.tight_layout()
    plt.savefig(modelPath+"cm {}.png".format(epoch_i))


# Function to use in order to test the model with any dataloader
def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Try to get the gpu to work instead of the cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs


def main(save_model, train_whole):

    # TRAINING
    set_seed(42)  # Set seed for reproducibility

    configuration = {
        "epochs": tune.choice([2, 3, 4, 5]),
        "epsilon": tune.choice([1e-8, 1e-6]),
        "learning_rate": tune.choice([1e-5, 2e-5, 3e-5, 4e-5, 5e-5]),
        "batch_size": tune.choice([16, 32, 64]),
        "betas": tune.choice([(0.9, 0.999), (0.9, 0.98)]),
    }

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "f1-score", "training_iteration"])

    result = tune.run(train_model_hyperparams,
                      resources_per_trial={"gpu": 1},
                      config=configuration,
                      num_samples=1,
                      progress_reporter=reporter)

    result.results_df.to_csv(modelPath + "results_df.csv")

    best_trial = result.get_best_trial(metric="accuracy", mode="max")

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # Now we train again with the best hyperparameters, to gather data and produce
    # some graphics ot evaluate the model
    train_dataloader, val_dataloader, y_true, category_sexism = data_loaders(best_trial.config)

    bert_classifier = train_model(best_trial.config, train_dataloader, val_dataloader, category_sexism)

    # TRAIN THE MODEL WITH THE WHOLE DATASET
    # if train_whole:
    #     # Concatenate the train_model set and the validation set
    #     full_train_data = torch.utils.data.ConcatDataset([train_data, val_data])
    #     full_train_sampler = RandomSampler(full_train_data)
    #     full_train_dataloader = DataLoader(full_train_data, sampler=full_train_sampler, batch_size=32)
    #
    #     # Train the Bert Classifier on the entire training data
    #     set_seed(42)
    #     bert_classifier, optimizer, scheduler = initialize_model(epochs=2)
    #     train_model(bert_classifier, full_train_dataloader, epochs=2)
    #
    # if save_model:
    #     # Save the model
    #     bert_classifier.save(modelPath)


if __name__ == "__main__":
    main(save_model=False, train_whole=False)

# Predictions on test set
# I have to change this to just predict the english part of the test dataset,
# while writing the results in a file formatted for submission.
"""test_data.sample(5)

#We also apply the preprocessing to the test set.


# Run `preprocessing_for_bert` on the test set
print('Tokenizing data...')
test_inputs, test_masks = preprocessing_for_bert(test_data.tweet)

# Create the DataLoader for our test set
test_dataset = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

# Predictions
# Compute predicted probabilities on the test set
probs = bert_predict(bert_classifier, test_dataloader)

# Get predictions from the probabilities
threshold = 0.9
preds = np.where(probs[:, 1] > threshold, 1, 0)

# Number of tweets predicted non-negative
print("Number of tweets predicted non-negative: ", preds.sum())

output = test_data[preds==1]
list(output.sample(20).tweet) """
