import pandas as pd
import torch
import torch.optim as optim
from prepared_input import Prepared
import deepmet_model
import time

batch_size = 8
epochs = 3
learning_rate = 0.00001
metaphor_preference_parameter = 0.2


# Loss function as defined by equations (7)-(9) in the paper
# is_verb_task: True = VERB track, False = ALLPOS track
def loss_function(estimate_metaphors, estimate_literals, targets, is_verb_task):
    l0 = l1 = - torch.sum(targets * torch.log(estimate_metaphors) + (1 - targets) * torch.log(estimate_literals))
    return l0 * int(is_verb_task) + l1 * (1 - int(is_verb_task))


def train(train_dataset, model, optimizer, epoch, is_verb_task):
    model.train()
    start = time.time()

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    for i_batch, sample_batched in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(*sample_batched[1:7])
        loss = loss_function(output[:, 1], output[:, 0], sample_batched[0], is_verb_task)
        loss.backward()
        optimizer.step()

        if i_batch % 1000 == 0:
            print("Epoch: " + str(epoch) +
                  (" Verb" if is_verb_task else " All Pos") +
                  ", Batch: " + str(i_batch) +
                  "/" + str(len(train_dataset) // batch_size + (1 if len(train_dataset) % batch_size > 0 else 0)) +
                  ", Time elapsed: " + str(time.time() - start))

    return model


def evaluate(eval_dataset, model):
    model.eval()
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

    start = time.time()

    probs = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):

            output = model(*sample_batched[1:7])
            probs.append(output)

            if i_batch % 500 == 0:
                print("Batch: " + str(i_batch) +
                      "/" + str(len(eval_dataset) // batch_size + (1 if len(eval_dataset) % batch_size > 0 else 0)) +
                      ", Time elapsed: " + str(time.time() - start))

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

    print("TP = " + str(tp) + ", FP = " + str(fp) + ", FN = " + str(fn) + ", TN = " + str(tn))

    acc = (tp + tn) / (tp + fp + fn + tn)
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (prec + rec) / 2

    print("Accuracy: " + str(acc))
    print("Precision: " + str(prec))
    print("Recall: " + str(rec))
    print("f1: " + str(f1))


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
    df_train_verb = pd.concat([df_train_vua_verb, df_train_toefl_verb])
    df_train_allpos = pd.concat([df_train_vua_allpos, df_train_toefl_allpos])

    train_verb_prepared = Prepared('train_verb', df_train_verb)
    train_allpos_prepared = Prepared('train_allpos', df_train_allpos)
    test_vua_verb_prepared = Prepared('test_vua_verb', df_test_vua_verb)
    test_vua_allpos_prepared = Prepared('test_vua_allpos', df_test_vua_allpos)
    test_toefl_verb_prepared = Prepared('test_toefl_verb', df_test_toefl_verb)
    test_toefl_allpos_prepared = Prepared('test_toefl_allpos', df_test_toefl_allpos)

    all_prepared = (train_verb_prepared, train_allpos_prepared, test_vua_verb_prepared, test_vua_allpos_prepared,
                    test_toefl_verb_prepared, test_toefl_allpos_prepared)

    # Make sure all inputs are of the same length
    max_length = max([prepared.length for prepared in all_prepared])
    for prepared in all_prepared:
        if prepared.length != max_length:
            prepared.prepare_inputs(max_length)

    model = deepmet_model.DeepMet(num_tokens=max_length,
                                  num_encoder_inputs=768,
                                  num_encoder_heads=2,
                                  num_encoder_layers=2,
                                  num_encoder_hidden_states=768,
                                  dropout_rate=0.2)

    if torch.cuda.is_available():
        for prepared in all_prepared:
            prepared.to_device(torch.device(0))
        model.to(torch.device(0))

    # 0 = metaphor, 1 = input_ids_a, 2 = att_mask_a, 3 = tok_type_ids_a, 4 = input_ids_b, 5 = att_mask_b,
    # 6 = tok_type_ids_b
    train_verb_dataset = torch.utils.data.TensorDataset(*train_verb_prepared.get_tensors())
    train_allpos_dataset = torch.utils.data.TensorDataset(*train_allpos_prepared.get_tensors())
    test_vua_verb_dataset = torch.utils.data.TensorDataset(*test_vua_verb_prepared.get_tensors())
    test_vua_allpos_dataset = torch.utils.data.TensorDataset(*test_vua_allpos_prepared.get_tensors())
    test_toefl_verb_dataset = torch.utils.data.TensorDataset(*test_toefl_verb_prepared.get_tensors())
    test_toefl_allpos_dataset = torch.utils.data.TensorDataset(*test_toefl_allpos_prepared.get_tensors())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Entering train")
    for epoch in range(epochs):
        model = train(train_verb_dataset, model, optimizer, epoch, True)
        model = train(train_allpos_dataset, model, optimizer, epoch, False)
    print("Done training!")

    torch.save(model.state_dict(), "../data/deepmet_model_1_1.model")
    print("Model saved!")

    # model.load_state_dict(torch.load("../data/deepmet_model_1.model"))

    print()
    print("VUA Verb evaluate")
    evaluate(test_vua_verb_dataset, model)
    print()
    print("VUA All POS evaluate")
    evaluate(test_vua_allpos_dataset, model)
    print()
    print("TOEFL Verb evaluate")
    evaluate(test_toefl_verb_dataset, model)
    print()
    print("TOEFL All POS evaluate")
    evaluate(test_toefl_allpos_dataset, model)
    print()


if __name__ == '__main__':
    main()
