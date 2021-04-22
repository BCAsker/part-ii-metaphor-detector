import pandas as pd
import random
import os
from constants import *

#num_folds = 10
vua_train_tokenized_file = "../data/VUA/tokenized/train_vua_tokenized.csv"
toefl_train_tokenized_file = "../data/TOEFL/tokenized/train_toefl_tokenized.csv"


def generate_stratified_folds(df):
    metaphor_indices = list(df[df['metaphor'] == 1].index)
    literal_indices = list(df[df['metaphor'] == 0].index)

    num_met_in_fold = len(metaphor_indices) // num_folds
    num_lit_in_fold = len(literal_indices) // num_folds

    met_remaining = len(metaphor_indices) % num_folds
    lit_remaining = len(literal_indices) % num_folds

    folds = [[] for _ in range(num_folds)]
    for i in range(num_folds):
        k_met = num_met_in_fold
        if met_remaining > 0:
            k_met += 1
            met_remaining -= 1
        fold_metaphors = random.sample(metaphor_indices, k=k_met)
        folds[i].extend(fold_metaphors)
        metaphor_indices = [index for index in metaphor_indices if index not in fold_metaphors]

        k_lit = num_lit_in_fold
        if lit_remaining > 0:
            k_lit += 1
            lit_remaining -= 1
        fold_literals = random.sample(literal_indices, k=k_lit)
        folds[num_folds-i-1].extend(fold_literals)  # Reverse index to better balance fold size
        literal_indices = [index for index in literal_indices if index not in fold_literals]

    return folds


def add_folds_to_dataframe(folds, df):
    df['fold'] = None
    for i, fold in enumerate(folds):
        df.loc[fold, 'fold'] = i
    return df


def check_folds_files_saved():
    return os.path.exists("../data/vua_train_folds.csv") and os.path.exists("../data/toefl_train_folds.csv")


def generate_folds():
    # df_vua = pd.read_csv(vua_train_tokenized_file, header=None, names=['id'])
    df_vua_labels = pd.read_csv("../data/VUA/tokenized/train_vua_tokenized.csv", index_col='token_id',
                                usecols=['token_id', 'metaphor'])
    df_toefl_labels = pd.read_csv("../data/TOEFL/tokenized/train_toefl_tokenized.csv", index_col='token_id',
                                  usecols=['token_id', 'metaphor'])

    vua_folds = generate_stratified_folds(df_vua_labels)
    df_vua_folds = add_folds_to_dataframe(vua_folds, df_vua_labels)

    toefl_folds = generate_stratified_folds(df_toefl_labels)
    df_toefl_folds = add_folds_to_dataframe(toefl_folds, df_toefl_labels)

    df_vua_folds.to_csv("../data/vua_train_folds.csv", columns=['fold'])
    df_toefl_folds.to_csv("../data/toefl_train_folds.csv", columns=['fold'])


if __name__ == '__main__':
    generate_folds()
