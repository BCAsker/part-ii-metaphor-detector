import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from prepared_input import Prepared
import deepmet_model
import time

batch_size = 16
max_seq_len = 128
epochs = 3
n_folds = 10
learning_rate = 0.00001
metaphor_preference_parameter = 0.2


# Loss function as defined by equation (7) in the paper. (8) and (9) redundant because we train in multi-task mode
def loss_function(estimate_metaphors, estimate_literals, targets):
    return - torch.sum(targets * torch.log(estimate_metaphors) + (1 - targets) * torch.log(estimate_literals))


def train(train_dataset, model, optimizer, epoch, model_num, start_time):
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
            print(f"Model number: {model_num}, "
                  f"Epoch: {epoch}, "
                  f"Batch: {i_batch}/"
                  f"{len(train_dataset) // batch_size + (1 if len(train_dataset) % batch_size > 0 else 0)}, "
                  f"Time elapsed: {(time.time() - start_time):.1f}s, "
                  f"Loss: {running_loss:.3f}")
            running_loss = 0.0

    return model


def evaluate(eval_dataset, models, start_time):
    if type(models) == deepmet_model.DeepMet:
        models = [models]
    [model.eval() for model in models]
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

    predictions = torch.zeros(len(eval_dataset), device=torch.device(0) if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for i_model in range(len(models)):
            if torch.cuda.is_available():
                models[i_model] = models[i_model].to(torch.device(0))

            probabilities = []
            for i_batch, sample_batched in enumerate(dataloader):
                output = (models[i_model])(*(sample_batched[1:7]))
                probabilities.append(output)
                if i_batch % 250 == 0:
                    print(f"Model number: {i_model}, "
                          f"Batch: {i_batch}/"
                          f"{len(eval_dataset) // batch_size + (1 if len(eval_dataset) % batch_size > 0 else 0)}, "
                          f"Time elapsed: {(time.time() - start_time):.1f}s")

            # Stitch together the predictions for each batch into a single tensor
            probabilities = torch.cat(probabilities)
            predictions += torch.argmax(probabilities, dim=1)

            # Free up GPU memory
            if torch.cuda.is_available():
                models[i_model] = models[i_model].cpu()

    predictions = predictions / len(models) >= metaphor_preference_parameter
    actuals = eval_dataset[:][0].tolist()

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for prediction, actual in zip(predictions, actuals):
        if prediction == actual == 1:
            tp += 1
        elif prediction == 1 and actual == 0:
            fp += 1
        elif prediction == 0 and actual == 1:
            fn += 1
        else:
            tn += 1

    print(f"TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")

    acc = (tp + tn) / (tp + fp + fn + tn)
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (prec + rec) / 2

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"f1: {f1:.3f}")

    return f1


def main():
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.width", 0)

    # Load the dataframes containing the raw inputs for our embedding layer
    df_train_vua = pd.read_csv("../data/VUA/train_vua_allpos_tokenized.csv", index_col='token_id').dropna()
    df_test_vua_verb = pd.read_csv("../data/VUA/test_vua_verb_tokenized.csv", index_col='token_id').dropna()
    df_test_vua_allpos = pd.read_csv("../data/VUA/test_vua_allpos_tokenized.csv", index_col='token_id').dropna()
    df_train_toefl = pd.read_csv("../data/TOEFL/train_toefl_allpos_tokenized.csv", index_col='token_id').dropna()
    df_test_toefl_verb = pd.read_csv("../data/TOEFL/test_toefl_verb_tokenized.csv", index_col='token_id').dropna()
    df_test_toefl_allpos = pd.read_csv("../data/TOEFL/test_toefl_allpos_tokenized.csv", index_col='token_id').dropna()

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

    if torch.cuda.is_available():
        train_vua_prepared.to_device(torch.device(0))
    train_vua_datasets = [torch.utils.data.TensorDataset(*tensors) for tensors in train_vua_prepared.to_folds(n_folds)]

    print("Entering VUA train")
    start = time.time()
    vua_models = []
    for i in range(n_folds):
        model = deepmet_model.DeepMet(num_tokens=max_length, dropout_rate=0.2)
        if torch.cuda.is_available():
            model = model.to(torch.device(0))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_dataset = torch.utils.data.ConcatDataset(train_vua_datasets[0:i] + train_vua_datasets[i+1:n_folds])
        eval_dataset = train_vua_datasets[i]

        for epoch in range(1, epochs + 1):
            model = train(train_dataset, model, optimizer, epoch, i, start)

        evaluate(eval_dataset, model, start)
        print()

        torch.save(model.state_dict(), f"../models/deepmet_model_VUA_2_{i}.model")
        if torch.cuda.is_available():
            model = model.cpu()
        vua_models.append(model)

    print("VUA train complete")

    if torch.cuda.is_available():
        train_vua_prepared.to_device('cpu')
        del train_vua_datasets
        train_toefl_prepared.to_device(torch.device(0))

    train_toefl_datasets = [torch.utils.data.TensorDataset(*tensors) for tensors in
                            train_toefl_prepared.to_folds(n_folds)]

    print("Entering TOEFL train")
    toefl_models = []
    for i in range(n_folds):
        model = deepmet_model.DeepMet(num_tokens=max_length, dropout_rate=0.2)
        if torch.cuda.is_available():
            model = model.to(torch.device(0))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_dataset = torch.utils.data.ConcatDataset(train_toefl_datasets[0:i] + train_toefl_datasets[i+1:n_folds])
        eval_dataset = train_toefl_datasets[i]

        for epoch in range(1, epochs + 1):
            model = train(train_dataset, model, optimizer, epoch, i, start)

        evaluate(eval_dataset, model, start)
        print()

        torch.save(model.state_dict(), f"../models/deepmet_model_TOEFL_2_{i}.model")
        if torch.cuda.is_available():
            model = model.cpu()
        toefl_models.append(model)

    print("Done training!")

    if torch.cuda.is_available():
        train_toefl_prepared.to_device('cpu')
        del train_toefl_datasets
        test_vua_verb_prepared.to_device(torch.device(0))
        test_vua_allpos_prepared.to_device(torch.device(0))
        test_toefl_verb_prepared.to_device(torch.device(0))
        test_toefl_allpos_prepared.to_device(torch.device(0))

    test_vua_verb_dataset = torch.utils.data.TensorDataset(*test_vua_verb_prepared.get_tensors())
    test_vua_allpos_dataset = torch.utils.data.TensorDataset(*test_vua_allpos_prepared.get_tensors())
    test_toefl_verb_dataset = torch.utils.data.TensorDataset(*test_toefl_verb_prepared.get_tensors())
    test_toefl_allpos_dataset = torch.utils.data.TensorDataset(*test_toefl_allpos_prepared.get_tensors())

    print("VUA Verb multi_evaluate")
    evaluate(test_vua_verb_dataset, vua_models, start)
    print()
    print("VUA All POS multi_evaluate")
    evaluate(test_vua_allpos_dataset, vua_models, start)
    print()
    print("TOEFL Verb multi_evaluate")
    evaluate(test_toefl_verb_dataset, toefl_models, start)
    print()
    print("TOEFL All POS multi_evaluate")
    evaluate(test_toefl_allpos_dataset, toefl_models, start)
    print()


if __name__ == '__main__':
    main()
