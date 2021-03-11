import sqlite3
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import SequentialSampler, DataLoader, TensorDataset, RandomSampler
from tqdm import trange
from transformers import XLNetForSequenceClassification, XLNetTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db = sqlite3.connect(r'E:\reddit_dataset\reddit-comments-may-2015\database.sqlite')

    df = pd.read_sql_query("SELECT * FROM May2015 LIMIT 1000", db)
    df['index'] = df.index

    # It's highly recommended to download bert prtrained model first, then save them into local file
    # In this document, contain confg(txt) and weight(bin) files
    model_file_address = 'model/'

    # Will load config and weight with from_pretrained()
    # Recommand download the model before using
    # Download model from "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-pytorch_model.bin"
    # Download model from "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-config.json"
    model = XLNetForSequenceClassification.from_pretrained(model_file_address, num_labels=2)

    sentences = df.body.values
    sentences = [sentence + " [SEP] [CLS]" for sentence in sentences]
    labels = df.controversiality.values
    print(sentences[199])

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    print("Tokenize the first sentence:")
    print(tokenized_texts[199])

    MAX_LEN = 128
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Use train_test_split to split our data into train and validation sets for training

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                        random_state=56, test_size=0.15)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                           random_state=56, test_size=0.15)

    # Convert all of our data into torch tensors, the required datatype for our model

    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Select a batch size for training. For fine-tuning with XLNet, the authors recommend a batch size of 32, 48, or 128. We will use 32 here to avoid memory issues.
    batch_size = 32

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=2e-5)

    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 4

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print(1)
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            print(2)
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            print(3)

            # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    torch.save(model.state_dict(),  '/model_with_language_model.ckpt')
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for i, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            # print (outputs)
            prediction = torch.argmax(outputs[0], dim=1)
            total += b_labels.size(0)
            correct += (prediction == b_labels).sum().item()
    print('Test Accuracy of the finetuned model on val data is: {} %'.format(100 * correct / total))



