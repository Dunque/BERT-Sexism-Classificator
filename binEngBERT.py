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
from sklearn.metrics import accuracy_score, roc_curve, auc
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
from ray.tune.schedulers import ASHAScheduler


configuration = {
    "epochs": random.choice([2, 3, 4, 5, 6]),
    "epsilon": random.choice([1e-8, 1e-6]),
    "learning_rate": random.choice([1e-5, 2e-5, 3e-5, 4e-5, 5e-5]),
    "batch_size": random.choice([16, 32, 64]),
    "betas": random.choice([(0.9, 0.999), (0.9, 0.98)])
}


# Function that returns two dataframes, for training and test
def load_data(translated_data='/home/roi_santos_rios/Desktop/BERT-Sexism-Classificator/data/EXIST2021_translatedTraining.csv',
              translated_test_data='/home/roi_santos_rios/Desktop/BERT-Sexism-Classificator/data/EXIST2021_translatedTest.csv'):
    # Load data and set labels
    data = pd.read_csv(translated_data)

    # convert labels to integers
    data['LabelTask1'] = data['task1'].apply(lambda x: 1 if x == 'sexist' else 0)

    # English version
    data = data[['id', 'English', 'LabelTask1']]
    data = data.rename(columns={'id': 'id', 'English': 'tweet', 'LabelTask1': 'label'})

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

    return data, test_data


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
        D_in, H, D_out = 768, 50, 2

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
def train_model(config, checkpoint_dir=None):
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
    data, test_data = load_data()

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
            os.path.join("models/binEngBert/checkpoints", "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

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

        with tune.checkpoint_dir(epoch_i) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(loss / val_step), accuracy=accuracy)

        # Compute the average accuracy and loss over the validation set.
        mean_loss = np.mean(val_loss)
        mean_accuracy = np.mean(val_accuracy)

        print(
            f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {mean_loss:^10.6f} | {mean_accuracy:^9.2f} | {time_elapsed:^9.2f}")
        print("-" * 70)

        print('Classification Report:')
        print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))

        print("\n")

        # if len(loss_list) >= 2:
        #     if loss_list[-2] < loss_list[-1]:
        #         print("Stopping training as loss started to grow")

    # if show_plots:
    #     bl = plt.subplot()
    #     bl.set_title('Batch loss')
    #     bl.set_xlabel('batch')
    #     bl.set_ylabel('loss')
    #     bl.plot(range(1, config["epochs"]+1), loss_list)
    #     plt.show()

    print("Training complete!")

    #return model


# EVALUATION
def evaluate(model, val_dataloader, avg_train_loss, time_elapsed, epoch_i, device, show_plots):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

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

    # Compute the average accuracy and loss over the validation set.
    mean_loss = np.mean(val_loss)
    mean_accuracy = np.mean(val_accuracy)

    print(
        f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {mean_loss:^10.6f} | {mean_accuracy:^9.2f} | {time_elapsed:^9.2f}")
    print("-" * 70)

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))

    if show_plots:
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
        ax.yaxis.set_ticklabels(['FAKE', 'REAL'])
        plt.show()

    return mean_loss, mean_accuracy


# EVALUATION on validation set
def bert_predict(model, test_dataloader, device):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
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


# Evaluation on validation set
def evaluate_roc(probs, y_true, show_plots):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')

    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Plot ROC AUC
    if show_plots:
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


def main(train_whole=False, save_model=False, show_plots=False):

    train_whole = train_whole
    save_model = save_model
    show_plots = show_plots

    # TRAINING
    # Actual training
    set_seed(42)  # Set seed for reproducibility
    # model = train_model(config, data, test_data, device, show_plots, evaluation=True)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(train_model,
                      resources_per_trial={"gpu": 1},
                      config=configuration,
                      num_samples=10,
                      progress_reporter=reporter)

    # # Compute predicted probabilities on the test set
    # probs = bert_predict(model, val_dataloader)
    #
    # # Evaluate the Bert classifier
    # evaluate_roc(probs, y_val, show_plots)
    #
    # # TRAIN THE MODEL WITH THE WHOLE DATASET
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
    #     model.save("models/binEngBERT")


if __name__ == "__main__":
    main(False, False, False)

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
