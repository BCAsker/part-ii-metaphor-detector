import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from prepared_input import Prepared
import deepmet_model
import time
import argparse

vua_fold_file = "../data/vua_train_folds.csv"
toefl_fold_file = "../data/toefl_train_folds.csv"


# Loss function as defined by equation (7) in the paper. (8) and (9) redundant because we train in multi-task mode
def loss_function(estimate_metaphors, estimate_literals, targets):
    return - torch.sum(targets * torch.log(estimate_metaphors) + (1 - targets) * torch.log(estimate_literals))


def train(train_dataset, model, optimizer, batch_size, epoch, model_num, start_time):
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


def evaluate(eval_dataset, models, batch_size, metaphor_preference_parameter, start_time, model_num=None):
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
                    print(f"Model number: {i_model if model_num is None else model_num}, "
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


def main(batch_size=16,
         max_seq_len=128,
         epochs=3, n_folds=10,
         learning_rate=0.00001,
         metaphor_preference_param=0.2,
         models_to_train=None):

    # Load the dataframes containing the raw inputs for our embedding layer. keep_default_na=False because nan might be
    # a valid string, although it should probably be Nan when referring to someone's grandmother
    df_train_vua = pd.read_csv("../data/VUA/tokenized/train_vua_tokenized.csv",
                               index_col='token_id', na_values=None, keep_default_na=False)
    df_test_vua_verb = pd.read_csv("../data/VUA/tokenized/test_vua_verb_tokenized.csv",
                                   index_col='token_id', keep_default_na=False)
    df_test_vua_allpos = pd.read_csv("../data/VUA/tokenized/test_vua_allpos_tokenized.csv",
                                     index_col='token_id', keep_default_na=False)
    df_train_toefl = pd.read_csv("../data/TOEFL/tokenized/train_toefl_tokenized.csv",
                                 index_col='token_id', keep_default_na=False)
    df_test_toefl_verb = pd.read_csv("../data/TOEFL/tokenized/test_toefl_verb_tokenized.csv",
                                     index_col='token_id', keep_default_na=False)
    df_test_toefl_allpos = pd.read_csv("../data/TOEFL/tokenized/test_toefl_allpos_tokenized.csv",
                                       index_col='token_id', keep_default_na=False)

    df_train_vua_folds = pd.read_csv("../data/vua_train_folds.csv", index_col='token_id')
    df_train_toefl_folds = pd.read_csv("../data/toefl_train_folds.csv", index_col='token_id')

    test_vua_verb_prepared = Prepared('test_vua_verb', df_test_vua_verb, max_seq_len)
    test_vua_allpos_prepared = Prepared('test_vua_allpos', df_test_vua_allpos, max_seq_len)
    test_toefl_verb_prepared = Prepared('test_toefl_verb', df_test_toefl_verb, max_seq_len)
    test_toefl_allpos_prepared = Prepared('test_toefl_allpos', df_test_toefl_allpos, max_seq_len)

    if models_to_train is None:
        models_to_train = list(range(n_folds))

    print("Entering VUA train")
    start = time.time()
    vua_models = []
    train_vua_prepared_folds = []
    train_vua_datasets = []
    for i in range(n_folds):
        fold_index = list(df_train_vua_folds[df_train_vua_folds['fold'] == i].index)
        prepared_fold = Prepared(f"train_vua_{i}", df_train_vua.loc[fold_index], max_seq_len)
        if torch.cuda.is_available():
            prepared_fold.to_device(torch.device(0))
        train_vua_prepared_folds.append(prepared_fold)
        train_vua_datasets.append(torch.utils.data.TensorDataset(*prepared_fold.get_tensors()))

    for i in models_to_train:
        model = deepmet_model.DeepMet(num_tokens=max_seq_len, dropout_rate=0.2)
        if torch.cuda.is_available():
            model = model.to(torch.device(0))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_dataset = torch.utils.data.ConcatDataset(train_vua_datasets[0:i] + train_vua_datasets[i+1:n_folds])
        eval_dataset = train_vua_datasets[i]

        for epoch in range(1, epochs + 1):
            model = train(train_dataset=train_dataset,
                          model=model,
                          optimizer=optimizer,
                          batch_size=batch_size,
                          epoch=epoch,
                          model_num=i,
                          start_time=start)
        print("Cross validation")
        evaluate(eval_dataset=eval_dataset,
                 models=model,
                 batch_size=batch_size,
                 metaphor_preference_parameter=metaphor_preference_param,
                 model_num=i,
                 start_time=start)
        print()

        torch.save(model.state_dict(), f"../models/deepmet_model_VUA_2_{i}.model")
        if torch.cuda.is_available():
            model = model.cpu()
        vua_models.append(model)

    # Cleanup GPU memory
    if torch.cuda.is_available():
        for prepared in train_vua_prepared_folds:
            prepared.to_device('cpu')
        del train_vua_prepared_folds
        del train_vua_datasets

    print("VUA train complete")

    print("Entering TOEFL train")
    toefl_models = []
    train_toefl_prepared_folds = []
    train_toefl_datasets = []
    for i in range(n_folds):
        fold_index = list(df_train_toefl_folds[df_train_toefl_folds['fold'] == i].index)
        prepared_fold = Prepared(f"train_toefl_{i}", df_train_toefl.loc[fold_index], max_seq_len)
        if torch.cuda.is_available():
            prepared_fold.to_device(torch.device(0))
        train_toefl_prepared_folds.append(prepared_fold)
        train_toefl_datasets.append(torch.utils.data.TensorDataset(*prepared_fold.get_tensors()))

    for i in models_to_train:
        model = deepmet_model.DeepMet(num_tokens=max_seq_len, dropout_rate=0.2)
        if torch.cuda.is_available():
            model = model.to(torch.device(0))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_dataset = torch.utils.data.ConcatDataset(train_toefl_datasets[0:i] + train_toefl_datasets[i+1:n_folds])
        eval_dataset = train_toefl_datasets[i]

        for epoch in range(1, epochs + 1):
            model = train(train_dataset=train_dataset,
                          model=model,
                          optimizer=optimizer,
                          batch_size=batch_size,
                          epoch=epoch,
                          model_num=i,
                          start_time=start)
        print("Cross validation")
        evaluate(eval_dataset=eval_dataset,
                 models=model,
                 batch_size=batch_size,
                 metaphor_preference_parameter=metaphor_preference_param,
                 model_num=i,
                 start_time=start)
        print()

        torch.save(model.state_dict(), f"../models/deepmet_model_TOEFL_2_{i}.model")
        if torch.cuda.is_available():
            model = model.cpu()
        toefl_models.append(model)

    # Cleanup GPU memory
    if torch.cuda.is_available():
        for prepared in train_toefl_prepared_folds:
            prepared.to_device('cpu')
        del train_toefl_prepared_folds
        del train_toefl_datasets

    print("TOEFL train complete")

    start = time.time()
    if torch.cuda.is_available():
        test_vua_verb_prepared.to_device(torch.device(0))
        test_vua_allpos_prepared.to_device(torch.device(0))
        test_toefl_verb_prepared.to_device(torch.device(0))
        test_toefl_allpos_prepared.to_device(torch.device(0))

    test_vua_verb_dataset = torch.utils.data.TensorDataset(*test_vua_verb_prepared.get_tensors())
    test_vua_allpos_dataset = torch.utils.data.TensorDataset(*test_vua_allpos_prepared.get_tensors())
    test_toefl_verb_dataset = torch.utils.data.TensorDataset(*test_toefl_verb_prepared.get_tensors())
    test_toefl_allpos_dataset = torch.utils.data.TensorDataset(*test_toefl_allpos_prepared.get_tensors())

    print("VUA Verb evaluation on test set")
    evaluate(eval_dataset=test_vua_verb_dataset,
             models=vua_models,
             batch_size=batch_size,
             metaphor_preference_parameter=metaphor_preference_param,
             start_time=start,
             model_num=None)
    print()
    print("VUA All POS evaluation on test set")
    evaluate(eval_dataset=test_vua_allpos_dataset,
             models=vua_models,
             batch_size=batch_size,
             metaphor_preference_parameter=metaphor_preference_param,
             start_time=start,
             model_num=None)
    print()
    print("TOEFL Verb evaluation on test set")
    evaluate(eval_dataset=test_toefl_verb_dataset,
             models=toefl_models,
             batch_size=batch_size,
             metaphor_preference_parameter=metaphor_preference_param,
             start_time=start,
             model_num=None)
    print()
    print("TOEFL All POS evaluation on test set")
    evaluate(eval_dataset=test_toefl_allpos_dataset,
             models=toefl_models,
             batch_size=batch_size,
             metaphor_preference_parameter=metaphor_preference_param,
             start_time=start,
             model_num=None)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepMet")
    parser.add_argument("--batch_size", default=16, required=False)
    parser.add_argument("--max_seq_len", default=128, required=False)
    parser.add_argument("--epochs", default=3, required=False)
    parser.add_argument("--folds", default=10, required=False)
    parser.add_argument("--learning_rate", default=0.00001, required=False)
    parser.add_argument("--metaphor_preference_param", default=0.2, required=False)
    parser.add_argument("--models_to_train", default=None, nargs="+", type=int,
                        help="Indices of the models to be trained")

    args = parser.parse_args()

    main(batch_size=args.batch_size,
         max_seq_len=args.max_seq_len,
         epochs=args.epochs,
         n_folds=args.folds,
         learning_rate=args.learning_rate,
         metaphor_preference_param=args.metaphor_preference_param,
         models_to_train=args.models_to_train)
