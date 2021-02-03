import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import preprocessing
import generate_training_folds
from prepared_input import Prepared
from deepmet_model import DeepMet
from constants import *
import time
import argparse


# Loss function as defined by equation (7) in the paper. (8) and (9) redundant because we train in multi-task mode
def loss_function(estimate_metaphors, estimate_literals, targets):
    return - torch.sum(targets * torch.log(estimate_metaphors) + (1 - targets) * torch.log(estimate_literals))


def train_models(dataset_name, df_train, df_folds, models_to_train, experiment_number, do_cross_eval, start_time,
                 colab_mode):
    models = []
    datasets = []

    for i in range(num_folds):
        fold_index = list(df_folds[df_folds['fold'] == i].index)
        prepared_fold = Prepared(f"train_{dataset_name}_{i}", df_train.loc[fold_index], max_seq_len)
        datasets.append(torch.utils.data.TensorDataset(*prepared_fold.get_tensors()))

    if colab_mode:
        scaler = torch.cuda.amp.GradScaler()
        local_size = 4 * batch_size
    else:
        scaler = None
        local_size = batch_size

    for i in models_to_train:
        model = DeepMet(num_tokens=max_seq_len, dropout_rate=dropout_rate)
        if torch.cuda.is_available():
            model = model.to(torch.device(0))
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_dataset = torch.utils.data.ConcatDataset(datasets[0:i] + datasets[i+1:num_folds])
        eval_dataset = datasets[i]

        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=local_size,
                                                     shuffle=True,
                                                     pin_memory=True,
                                                     num_workers=4)

            for i_batch, sample_batched in enumerate(dataloader):
                if torch.cuda.is_available():
                    sample_batched = [s.to(torch.device(0), non_blocking=True) for s in sample_batched]
                optimizer.zero_grad()
                if colab_mode:
                    with torch.cuda.amp.autocast():
                        output = model(*sample_batched[1:4])
                        loss = loss_function(output[:, 1], output[:, 0], sample_batched[0])
                    running_loss += float(loss)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(*sample_batched[1:4])
                    loss = loss_function(output[:, 1], output[:, 0], sample_batched[0])
                    running_loss += float(loss)
                    loss.backward()
                    optimizer.step()

                if i_batch % 500 == 0:
                    print(f"Model number: {i}, "
                          f"Epoch: {epoch}, "
                          f"Batch: {i_batch}/"
                          f"{len(train_dataset) // local_size + (1 if len(train_dataset) % local_size > 0 else 0)}, "
                          f"Time elapsed: {(time.time() - start_time):.1f}s, "
                          f"Loss: {running_loss:.3f}")
                    running_loss = 0.0

        if do_cross_eval:
            print("Cross validation")
            evaluate(eval_dataset=eval_dataset,
                     models=model,
                     model_num=i,
                     start_time=start_time)
        print()

        torch.save(model.state_dict(),
                   f"../models/{experiment_number}/deepmet_model_{dataset_name}_{experiment_number}_{i}.model")
        if torch.cuda.is_available():
            model = model.cpu()
        models.append(model)

    return models


def load_trained_models(dataset_name, models_to_load, experiment_number, do_cross_eval, df_train, df_folds, start):
    models = []
    cross_eval_datasets = []

    if do_cross_eval:
        for i in range(num_folds):
            fold_index = list(df_folds[df_folds['fold'] == i].index)
            prepared_fold = Prepared(f"train_VUA_{i}", df_train.loc[fold_index], max_seq_len)
            cross_eval_datasets.append(torch.utils.data.TensorDataset(*prepared_fold.get_tensors()))
        print("Cross validation")

    for i in models_to_load:
        model = DeepMet(num_tokens=max_seq_len, dropout_rate=dropout_rate)
        model.load_state_dict(torch.load(
            f"../models/{experiment_number}/deepmet_model_{dataset_name}_{experiment_number}_{i}.model"))

        if do_cross_eval:
            evaluate(eval_dataset=cross_eval_datasets[i],
                     models=model,
                     model_num=i,
                     start_time=start)
            print()

        models.append(model)

    return models


def evaluate(eval_dataset, models, start_time, model_num=None):
    if type(models) == DeepMet:
        models = [models]
    [model.eval() for model in models]
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)
    predictions = torch.zeros(len(eval_dataset), device=torch.device(0) if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for i_model in range(len(models)):
            if torch.cuda.is_available():
                models[i_model] = models[i_model].to(torch.device(0))

            probabilities = []
            for i_batch, sample_batched in enumerate(dataloader):
                if torch.cuda.is_available():
                    sample_batched = [s.to(torch.device(0), non_blocking=True) for s in sample_batched]
                output = (models[i_model])(*(sample_batched[1:4]))
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


def main(use_vua=True, use_toefl=True, load_saved=False, do_cross_eval=False, do_final_eval=False, model_indices=None,
         expr_num=0, colab_mode=False):

    start = time.time()
    if model_indices is None:
        model_indices = list(range(num_folds))

    if not preprocessing.check_preprocessed_files_saved():
        print("Preparing data triples")
        preprocessing.initial_preprocessing()
        print("Done preparing data triples")
        print()

    if not generate_training_folds.check_folds_files_saved():
        print("Generating training folds")
        generate_training_folds.generate_folds()
        print("Training folds generated")
        print()

    vua_models = []
    toefl_models = []

    if use_vua:
        df_train_vua = pd.read_csv("../data/VUA/tokenized/train_vua_tokenized.csv", index_col='token_id',
                                   na_values=None, keep_default_na=False)
        df_train_vua_folds = pd.read_csv("../data/vua_train_folds.csv", index_col='token_id')

        if load_saved:
            print("Loading trained VUA models")
            vua_models = load_trained_models(dataset_name="VUA", models_to_load=model_indices,
                                             experiment_number=expr_num, do_cross_eval=do_cross_eval,
                                             df_train=df_train_vua, df_folds=df_train_vua_folds, start=start)
            print("Finished loading trained VUA models")
        else:
            print("Entering VUA train")
            vua_models = train_models(dataset_name="VUA", df_train=df_train_vua, df_folds=df_train_vua_folds,
                                      models_to_train=model_indices, experiment_number=expr_num,
                                      do_cross_eval=do_cross_eval,
                                      start_time=start, colab_mode=colab_mode)
            print("VUA train complete")

    if use_toefl:
        df_train_toefl = pd.read_csv("../data/TOEFL/tokenized/train_toefl_tokenized.csv",
                                     index_col='token_id', keep_default_na=False)
        df_train_toefl_folds = pd.read_csv("../data/toefl_train_folds.csv", index_col='token_id')

        if load_saved:
            print("Loading trained TOEFL models")
            toefl_models = load_trained_models(dataset_name="TOEFL", models_to_load=model_indices,
                                               experiment_number=expr_num, do_cross_eval=do_cross_eval,
                                               df_train=df_train_toefl, df_folds=df_train_toefl_folds, start=start)
            print("Finished loading trained TOEFL models")
        else:
            print("Entering TOEFL train")
            toefl_models = train_models(dataset_name="TOEFL", df_train=df_train_toefl, df_folds=df_train_toefl_folds,
                                        models_to_train=model_indices, experiment_number=expr_num,
                                        do_cross_eval=do_cross_eval,
                                        start_time=start, colab_mode=colab_mode)
            print("TOEFL train complete")

    if do_final_eval:
        if use_vua:
            df_test_vua_verb = pd.read_csv("../data/VUA/tokenized/test_vua_verb_tokenized.csv",
                                           index_col='token_id', keep_default_na=False)
            df_test_vua_allpos = pd.read_csv("../data/VUA/tokenized/test_vua_allpos_tokenized.csv",
                                             index_col='token_id', keep_default_na=False)
            test_vua_verb_prepared = Prepared('test_vua_verb', df_test_vua_verb, max_seq_len)
            test_vua_allpos_prepared = Prepared('test_vua_allpos', df_test_vua_allpos, max_seq_len)

            test_vua_verb_dataset = torch.utils.data.TensorDataset(*test_vua_verb_prepared.get_tensors())
            test_vua_allpos_dataset = torch.utils.data.TensorDataset(*test_vua_allpos_prepared.get_tensors())

            print("VUA Verb evaluation on test set")
            evaluate(eval_dataset=test_vua_verb_dataset,
                     models=vua_models,
                     start_time=start,
                     model_num=None)
            print()
            print("VUA All POS evaluation on test set")
            evaluate(eval_dataset=test_vua_allpos_dataset,
                     models=vua_models,
                     start_time=start,
                     model_num=None)
            print()

        if use_toefl:
            df_test_toefl_verb = pd.read_csv("../data/TOEFL/tokenized/test_toefl_verb_tokenized.csv",
                                             index_col='token_id', keep_default_na=False)
            df_test_toefl_allpos = pd.read_csv("../data/TOEFL/tokenized/test_toefl_allpos_tokenized.csv",
                                               index_col='token_id', keep_default_na=False)
            test_toefl_verb_prepared = Prepared('test_toefl_verb', df_test_toefl_verb, max_seq_len)
            test_toefl_allpos_prepared = Prepared('test_toefl_allpos', df_test_toefl_allpos, max_seq_len)

            test_toefl_verb_dataset = torch.utils.data.TensorDataset(*test_toefl_verb_prepared.get_tensors())
            test_toefl_allpos_dataset = torch.utils.data.TensorDataset(*test_toefl_allpos_prepared.get_tensors())

            print("TOEFL Verb evaluation on test set")
            evaluate(eval_dataset=test_toefl_verb_dataset,
                     models=toefl_models,
                     start_time=start,
                     model_num=None)
            print()
            print("TOEFL All POS evaluation on test set")
            evaluate(eval_dataset=test_toefl_allpos_dataset,
                     models=toefl_models,
                     start_time=start,
                     model_num=None)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepMet")
    parser.add_argument('--vua', dest='vua', action='store_true')
    parser.add_argument('--no_vua', dest='vua', action='store_false')

    parser.add_argument('--toefl', dest='toefl', action='store_true')
    parser.add_argument('--no_toefl', dest='toefl', action='store_false')

    parser.add_argument('--load_saved_models', dest='load_saved_models', action='store_true')
    parser.add_argument('--no_load_saved_models', dest='load_saved_models', action='store_false')

    parser.add_argument('--do_cross_eval', dest='do_cross_eval', action='store_true')
    parser.add_argument('--no_cross_eval', dest='do_cross_eval', action='store_false')

    parser.add_argument('--do_final_eval', dest='do_final_eval', action='store_true')
    parser.add_argument('--no_final_eval', dest='do_final_eval', action='store_false')

    parser.add_argument('--colab_mode', dest='colab_mode', action='store_true')
    parser.add_argument('--no_colab_mode', dest='colab_mode', action='store_false')

    parser.add_argument('--experiment_number', default=0, required=False, help="Number to be given to the saved models")

    parser.add_argument('--model_indices', default=None, nargs="+", type=int, required=False,
                        help="Indices of the models to be trained")

    parser.set_defaults(vua=True, toefl=True, load_saved_models=False, do_cross_eval=False, do_final_eval=False,
                        colab_mode=False)

    args = parser.parse_args()

    use_vua_arg = args.vua
    use_toefl_arg = args.toefl
    load_saved_models_arg = args.load_saved_models
    do_cross_eval_arg = args.do_cross_eval
    do_final_eval_arg = args.do_final_eval
    expr_num_arg = args.experiment_number
    model_indices_arg = args.model_indices
    colab_mode_arg = args.colab_mode

    main(use_vua=use_vua_arg,
         use_toefl=use_toefl_arg,
         load_saved=load_saved_models_arg,
         do_cross_eval=do_cross_eval_arg,
         do_final_eval=do_final_eval_arg,
         expr_num=expr_num_arg,
         model_indices=model_indices_arg,
         colab_mode=colab_mode_arg)

