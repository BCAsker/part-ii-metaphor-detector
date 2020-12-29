import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import os
import embeddings
import deepmet_model
import time

batch_size = 4
epochs = 3
learning_rate = 0.00001
metaphor_preference_parameter = 0.2


def save_train_prepared(global_input_ids, global_attention_mask, global_token_type_ids, local_input_ids, local_attention_mask, local_token_type_ids):
    torch.save(global_input_ids, '../data/intermediate/train_global_input_ids.pt')
    torch.save(global_attention_mask, '../data/intermediate/train_global_attention_mask.pt')
    torch.save(global_token_type_ids, '../data/intermediate/train_global_token_type_ids.pt')
    torch.save(local_input_ids, '../data/intermediate/train_local_input_ids.pt')
    torch.save(local_attention_mask, '../data/intermediate/train_local_attention_mask.pt')
    torch.save(local_token_type_ids, '../data/intermediate/train_local_token_type_ids.pt')


def load_train_prepared():
    global_input_ids = torch.load('../data/intermediate/train_global_input_ids.pt')
    global_attention_mask = torch.load('../data/intermediate/train_global_attention_mask.pt')
    global_token_type_ids = torch.load('../data/intermediate/train_global_token_type_ids.pt')
    local_input_ids = torch.load('../data/intermediate/train_local_input_ids.pt')
    local_attention_mask = torch.load('../data/intermediate/train_local_attention_mask.pt')
    local_token_type_ids = torch.load('../data/intermediate/train_local_token_type_ids.pt')
    return global_input_ids, global_attention_mask, global_token_type_ids, local_input_ids, local_attention_mask, local_token_type_ids


def save_test_verb_prepared(global_input_ids, global_attention_mask, global_token_type_ids, local_input_ids, local_attention_mask, local_token_type_ids):
    torch.save(global_input_ids, '../data/intermediate/test_verb_global_input_ids.pt')
    torch.save(global_attention_mask, '../data/intermediate/test_verb_global_attention_mask.pt')
    torch.save(global_token_type_ids, '../data/intermediate/test_verb_global_token_type_ids.pt')
    torch.save(local_input_ids, '../data/intermediate/test_verb_local_input_ids.pt')
    torch.save(local_attention_mask, '../data/intermediate/test_verb_local_attention_mask.pt')
    torch.save(local_token_type_ids, '../data/intermediate/test_verb_local_token_type_ids.pt')


def load_test_verb_prepared():
    global_input_ids = torch.load('../data/intermediate/test_verb_global_input_ids.pt')
    global_attention_mask = torch.load('../data/intermediate/test_verb_global_attention_mask.pt')
    global_token_type_ids = torch.load('../data/intermediate/test_verb_global_token_type_ids.pt')
    local_input_ids = torch.load('../data/intermediate/test_verb_local_input_ids.pt')
    local_attention_mask = torch.load('../data/intermediate/test_verb_local_attention_mask.pt')
    local_token_type_ids = torch.load('../data/intermediate/test_verb_local_token_type_ids.pt')
    return global_input_ids, global_attention_mask, global_token_type_ids, local_input_ids, local_attention_mask, local_token_type_ids


def save_test_allpos_prepared(global_input_ids, global_attention_mask, global_token_type_ids, local_input_ids, local_attention_mask, local_token_type_ids):
    torch.save(global_input_ids, '../data/intermediate/test_allpos_global_input_ids.pt')
    torch.save(global_attention_mask, '../data/intermediate/test_allpos_global_attention_mask.pt')
    torch.save(global_token_type_ids, '../data/intermediate/test_allpos_global_token_type_ids.pt')
    torch.save(local_input_ids, '../data/intermediate/test_allpos_local_input_ids.pt')
    torch.save(local_attention_mask, '../data/intermediate/test_allpos_local_attention_mask.pt')
    torch.save(local_token_type_ids, '../data/intermediate/test_allpos_local_token_type_ids.pt')


def load_test_allpos_prepared():
    global_input_ids = torch.load('../data/intermediate/test_allpos_global_input_ids.pt')
    global_attention_mask = torch.load('../data/intermediate/test_allpos_global_attention_mask.pt')
    global_token_type_ids = torch.load('../data/intermediate/test_allpos_global_token_type_ids.pt')
    local_input_ids = torch.load('../data/intermediate/test_allpos_local_input_ids.pt')
    local_attention_mask = torch.load('../data/intermediate/test_allpos_local_attention_mask.pt')
    local_token_type_ids = torch.load('../data/intermediate/test_allpos_local_token_type_ids.pt')
    return global_input_ids, global_attention_mask, global_token_type_ids, local_input_ids, local_attention_mask, local_token_type_ids


def check_train_prepared():
    names = ['train_global_input_ids.pt', 'train_global_attention_mask.pt', 'train_global_token_type_ids.pt', 'train_local_input_ids.pt', 'train_local_attention_mask.pt', 'train_local_token_type_ids.pt']
    return np.all([os.path.exists('../data/intermediate/' + name) for name in names])


def check_test_verb_prepared():
    names = ['test_verb_global_input_ids.pt', 'test_verb_global_attention_mask.pt', 'test_verb_global_token_type_ids.pt', 'test_verb_local_input_ids.pt', 'test_verb_local_attention_mask.pt', 'test_verb_local_token_type_ids.pt']
    return np.all([os.path.exists('../data/intermediate/' + name) for name in names])


def check_test_allpos_prepared():
    names = ['test_allpos_global_input_ids.pt', 'test_allpos_global_attention_mask.pt', 'test_allpos_global_token_type_ids.pt', 'test_allpos_local_input_ids.pt', 'test_allpos_local_attention_mask.pt', 'test_allpos_local_token_type_ids.pt']
    return np.all([os.path.exists('../data/intermediate/' + name) for name in names])


# Loss function as defined by equations (7)-(9) in the paper
# is_verb_task: True = VERB track, False = ALLPOS track
def loss_function(estimate_metaphors, estimate_literals, targets, is_verb_task):
    l0 = l1 = - torch.sum(targets * torch.log(estimate_metaphors) + (1 - targets) * torch.log(estimate_literals))
    return l0 * int(is_verb_task) + l1 * (1 - int(is_verb_task))


def train(train_dataset, model, optimizer, epoch):
    model.train()
    start = time.time()

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    for i_batch, sample_batched in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(*sample_batched[1:7])
        loss = loss_function(output[:, 1], output[:, 0], sample_batched[0], False)
        loss.backward()
        optimizer.step()

        if i_batch % 500 == 0:
            print("Epoch: " + str(epoch) +
                  ", Batch: " + str(i_batch) +
                  "/" + str(len(train_dataset) // batch_size + (1 if len(train_dataset) % batch_size > 0 else 0)) +
                  ", Time elapsed: " + str(time.time() - start))

    return model


def evaluate(eval_dataset, model, epoch):
    model.eval()
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

    start = time.time()

    actuals = []
    probs = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            output = model(*sample_batched[1:7])
            probs.append(output)
            actuals.extend(sample_batched[0])

            if i_batch % 500 == 0:
                print("Epoch: " + str(epoch) +
                      ", Batch: " + str(i_batch) +
                      "/" + str(len(eval_dataset) // batch_size + (1 if len(eval_dataset) % batch_size > 0 else 0)) +
                      ", Time elapsed: " + str(time.time() - start))

    preds = [int(prob > metaphor_preference_parameter) for prob in torch.cat(probs, dim=0)[:, 1]]

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    first = True

    for pred, actual in zip(preds, actuals):
        if first:
            print(pred)
            print(actual)
            first = False
        if pred == actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 1:
            fn += 1
        else:
            tn += 1

    acc = (tp + tn) / (tp + fp + fn + tn)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (prec + rec) / 2

    print("Accuracy: " + str(acc))
    print("Precision: " + str(prec))
    print("Recall: " + str(rec))
    print("f1: " + str(f1))


def main():
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.width", 0)

    # Load the dataframes containing the raw inputs for our embedding layer
    df_vua_train = pd.read_csv('../data/VUA/vua_train_tokenized.csv', index_col='token_id').dropna()
    df_toefl_train = pd.read_csv('../data/TOEFL/toefl_train_tokenized.csv', index_col='token_id').dropna()
    df_vua_test_verb = pd.read_csv('../data/VUA/vua_test_verb_tokenized.csv', index_col='token_id').dropna()
    df_toefl_test_verb = pd.read_csv('../data/TOEFL/toefl_test_verb_tokenized.csv', index_col='token_id').dropna()
    df_vua_test_allpos = pd.read_csv('../data/VUA/vua_test_allpos_tokenized.csv', index_col='token_id').dropna()
    df_toefl_test_allpos = pd.read_csv('../data/TOEFL/toefl_test_allpos_tokenized.csv', index_col='token_id').dropna()

    df_train = pd.concat([df_vua_train, df_toefl_train])
    df_test_verb = pd.concat([df_vua_test_verb, df_toefl_test_verb])
    df_test_allpos = pd.concat([df_vua_test_allpos, df_toefl_test_allpos])

    if check_train_prepared() and check_test_verb_prepared() and check_test_allpos_prepared():
        train_prepared = load_train_prepared()
        test_verb_prepared = load_test_verb_prepared()
        test_allpos_prepared = load_test_allpos_prepared()

        max_length = len(train_prepared[0][0])

    else:
        train_prepared = embeddings.prepare_inputs(df_train, 0)
        test_verb_prepared = embeddings.prepare_inputs(df_test_verb, 0)
        test_allpos_prepared = embeddings.prepare_inputs(df_test_allpos, 0)

        max_length = max(len(train_prepared[0][0]), len(test_verb_prepared[0][0]), len(test_allpos_prepared[0][0]))

        # Make sure all inputs are padded to the same length
        if len(train_prepared[0][0]) != max_length:
            train_prepared = embeddings.prepare_inputs(df_train, max_length)
        if len(test_verb_prepared[0][0]) != max_length:
            test_verb_prepared = embeddings.prepare_inputs(df_test_verb, max_length)
        if len(test_allpos_prepared[0][0]) != max_length:
            test_allpos_prepared = embeddings.prepare_inputs(df_test_allpos, max_length)

        save_train_prepared(*train_prepared)
        save_test_verb_prepared(*test_verb_prepared)
        save_test_allpos_prepared(*test_allpos_prepared)

    train_prepared = list(train_prepared)
    test_verb_prepared = list(test_verb_prepared)
    test_allpos_prepared = list(test_allpos_prepared)

    model = deepmet_model.DeepMet(num_tokens=max_length,
                                  num_encoder_inputs=768,
                                  num_encoder_heads=2,
                                  num_encoder_layers=2,
                                  num_encoder_hidden_states=768,
                                  dropout_rate=0.2)

    training_labels = torch.tensor(df_train['metaphor'], dtype=torch.int64)
    test_verb_labels = torch.tensor(df_test_verb['metaphor'], dtype=torch.int64)
    test_allpos_labels = torch.tensor(df_test_allpos['metaphor'], dtype=torch.int64)

    if torch.cuda.is_available():
        for i in range(len(train_prepared)):
            train_prepared[i] = (train_prepared[i]).to(torch.device(0))
            test_verb_prepared[i] = (test_verb_prepared[i]).to(torch.device(0))
            test_allpos_prepared[i] = (test_allpos_prepared[i]).to(torch.device(0))
        training_labels = training_labels.to(torch.device(0))
        model.to(torch.device(0))

    # 0 = metaphor, 1 = input_ids_a, 2 = att_mask_a, 3 = tok_type_ids_a, 4 = input_ids_b, 5 = att_mask_b,
    # 6 = tok_type_ids_b
    training_dataset = torch.utils.data.TensorDataset(training_labels, *train_prepared)
    verb_evaluation_dataset = torch.utils.data.TensorDataset(test_verb_labels, *test_verb_prepared)
    allpos_evaluation_dataset = torch.utils.data.TensorDataset(test_allpos_labels, *test_allpos_prepared)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Entering train")
    for epoch in range(epochs):
        model = train(training_dataset, model, optimizer, epoch)
    print("Done training!")

    torch.save(model.state_dict(), "../data/deepmet_one.model")
    print("Model saved!")

    print("Entering verb evaluate")
    evaluate(verb_evaluation_dataset, model, 1)
    print("Done!")
    print()
    print("Entering allpos evaluate")
    evaluate(allpos_evaluation_dataset, model, 1)
    print("Done!")


if __name__ == '__main__':
    main()
