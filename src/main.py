import pandas as pd
import torch
import torch.optim as optim
from prepared_input import Prepared
import deepmet_model
import time

batch_size = 16
max_seq_len = 128
epochs = 3
learning_rate = 0.00001
metaphor_preference_parameter = 0.2


# Loss function as defined by equation (7) in the paper. (8) and (9) redundant because we train in multi-task mode
def loss_function(estimate_metaphors, estimate_literals, targets):
    return - torch.sum(targets * torch.log(estimate_metaphors) + (1 - targets) * torch.log(estimate_literals))


def train(train_dataset, model, optimizer, epoch, start_time):
    model.train()

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    running_loss = 0.0
    for i_batch, sample_batched in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(*sample_batched[1:7])
        loss = loss_function(output[:, 1], output[:, 0], sample_batched[0])
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i_batch % 500 == 0:
            print(f"Epoch: {epoch}, "
                  f"Batch: {i_batch}/"
                  f"{len(train_dataset) // batch_size + (1 if len(train_dataset) % batch_size > 0 else 0)}, "
                  f"Time elapsed: {(time.time()-start_time):.1f}s, "
                  f"Loss: {running_loss:.3f}")
            running_loss = 0

    return model


def evaluate(eval_dataset, model):
    model.eval()
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

    start_time = time.time()

    probs = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):

            output = model(*sample_batched[1:7])
            probs.append(output)

            if i_batch % 250 == 0:
                print(f"Batch: {i_batch}/"
                      f"{len(eval_dataset) // batch_size + (1 if len(eval_dataset) % batch_size > 0 else 0)}, "
                      f"Time elapsed: {(time.time() - start_time):.1f}s")

    preds = [int(prob > metaphor_preference_parameter) for prob in torch.cat(probs, dim=0)[:, 1]]
    actuals = eval_dataset[:][0].tolist()

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for pred, actual in zip(preds, actuals):
        if pred == actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 1:
            fn += 1
        else:
            tn += 1

    print(f"TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")

    acc = (tp + tn) / (tp + fp + fn + tn)
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (prec + rec) / 2

    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"f1: {f1}")


def main():
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.width", 0)

    # Load the dataframes containing the raw inputs for our embedding layer
    df_train_vua_verb = pd.read_csv("../data/VUA/train_vua_verb_tokenized.csv", index_col='token_id').dropna()
    df_train_vua_allpos = pd.read_csv("../data/VUA/train_vua_allpos_tokenized.csv", index_col='token_id').dropna()
    df_test_vua_verb = pd.read_csv("../data/VUA/test_vua_verb_tokenized.csv", index_col='token_id').dropna()
    df_test_vua_allpos = pd.read_csv("../data/VUA/test_vua_allpos_tokenized.csv", index_col='token_id').dropna()
    df_train_toefl_verb = pd.read_csv("../data/TOEFL/train_toefl_verb_tokenized.csv", index_col='token_id').dropna()
    df_train_toefl_allpos = pd.read_csv("../data/TOEFL/train_toefl_allpos_tokenized.csv", index_col='token_id').dropna()
    df_test_toefl_verb = pd.read_csv("../data/TOEFL/test_toefl_verb_tokenized.csv", index_col='token_id').dropna()
    df_test_toefl_allpos = pd.read_csv("../data/TOEFL/test_toefl_allpos_tokenized.csv", index_col='token_id').dropna()

    # df_train_vua = pd.concat([df_train_vua_verb, df_train_vua_allpos])
    # df_train_toefl = pd.concat([df_train_toefl_verb, df_train_toefl_allpos])
    df_train_vua = df_train_vua_allpos
    df_train_toefl = df_train_toefl_allpos

    train_vua_prepared = Prepared('train_vua', df_train_vua, max_seq_len)
    train_toefl_prepared = Prepared('train_toefl', df_train_toefl, max_seq_len)
    test_vua_verb_prepared = Prepared('test_vua_verb', df_test_vua_verb, max_seq_len)
    test_vua_allpos_prepared = Prepared('test_vua_allpos', df_test_vua_allpos, max_seq_len)
    test_toefl_verb_prepared = Prepared('test_toefl_verb', df_test_toefl_verb, max_seq_len)
    test_toefl_allpos_prepared = Prepared('test_toefl_allpos', df_test_toefl_allpos, max_seq_len)

    all_prepared = (train_vua_prepared, train_toefl_prepared, test_vua_verb_prepared, test_vua_allpos_prepared,
                    test_toefl_verb_prepared, test_toefl_allpos_prepared)

    # Make sure all inputs are of the same length
    max_length = max([prepared.length for prepared in all_prepared])
    for prepared in all_prepared:
        if prepared.length != max_length:
            prepared.prepare_inputs(max_length)

    model = deepmet_model.DeepMet(num_tokens=max_length, dropout_rate=0.2)

    if torch.cuda.is_available():
        for prepared in all_prepared:
            prepared.to_device(torch.device(0))
        model.to(torch.device(0))

    # 0 = metaphor, 1 = input_ids_a, 2 = att_mask_a, 3 = tok_type_ids_a, 4 = input_ids_b, 5 = att_mask_b,
    # 6 = tok_type_ids_b
    train_vua_dataset = torch.utils.data.TensorDataset(*train_vua_prepared.get_tensors())
    train_toefl_dataset = torch.utils.data.TensorDataset(*train_toefl_prepared.get_tensors())
    test_vua_verb_dataset = torch.utils.data.TensorDataset(*test_vua_verb_prepared.get_tensors())
    test_vua_allpos_dataset = torch.utils.data.TensorDataset(*test_vua_allpos_prepared.get_tensors())
    test_toefl_verb_dataset = torch.utils.data.TensorDataset(*test_toefl_verb_prepared.get_tensors())
    test_toefl_allpos_dataset = torch.utils.data.TensorDataset(*test_toefl_allpos_prepared.get_tensors())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Entering TOEFL train")
    start = time.time()
    for epoch in range(epochs):
        model = train(train_toefl_dataset, model, optimizer, epoch, start)
    print("Done training TOEFL!")
    torch.save(model.state_dict(), "../models/deepmet_model_TOEFL_1_5.model")
    print("TOEFL Model saved!")

    print()
    print("TOEFL Verb evaluate")
    evaluate(test_toefl_verb_dataset, model)
    print()
    print("TOEFL All POS evaluate")
    evaluate(test_toefl_allpos_dataset, model)
    print()

    # Create a fresh model for the VUA tasks
    if torch.cuda.is_available():
        # Remove the TOEFL model from the GPU, so that we have enough room for the VUA one
        model = model.cpu()

    model = deepmet_model.DeepMet(num_tokens=max_length, dropout_rate=0.2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        model.to(torch.device(0))

    print("Entering VUA train")
    start = time.time()
    for epoch in range(epochs):
        model = train(train_vua_dataset, model, optimizer, epoch, start)
    print("Done training VUA!")
    torch.save(model.state_dict(), "../models/deepmet_model_VUA_1_5.model")
    print("VUA Model saved!")

    print()
    print("VUA Verb evaluate")
    evaluate(test_vua_verb_dataset, model)
    print()
    print("VUA All POS evaluate")
    evaluate(test_vua_allpos_dataset, model)


if __name__ == '__main__':
    main()
