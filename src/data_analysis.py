import numpy as np
import pandas as pd
from transformers import RobertaTokenizer
from constants import *


# Reduced version of prepare_inputs used to see how many rows of data will get truncated with the given max_seq_len
def prepare_inputs(df):
    # global_context_input = df['sentence'] + " </s> " + df['query'] + " </s> " + df['pos'] + " </s> " + df['fgpos']
    # local_context_input = df['local'] + " </s> " + df['query'] + " </s> " + df['pos'] + " </s> " + df['fgpos']
    global_context_input = df['sentence'] + " </s> " + df['query']
    local_context_input = df['local'] + " </s> " + df['query']

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    global_tokenized = tokenizer(list(global_context_input), add_special_tokens=True, return_tensors='np')['input_ids']
    local_tokenized = tokenizer(list(local_context_input), add_special_tokens=True, return_tensors='np')['input_ids']

    print("Global tokenized")
    longer = 0
    for row in global_tokenized:
        if len(row) > max_seq_len:
            longer += 1
    print(f"Longer: {longer}/{len(global_tokenized)}")

    print("Local tokenized")
    longer = 0
    for row in local_tokenized:
        if len(row) > max_seq_len:
            longer += 1
    print(f"Longer: {longer}/{len(local_tokenized)}")

def main():
    df_train_vua = pd.read_csv("../data/VUA/tokenized/train_vua_tokenized.csv", index_col='token_id', na_values=None,
                               keep_default_na=False)
    df_train_toefl = pd.read_csv("../data/TOEFL/tokenized/train_toefl_tokenized.csv", index_col='token_id',
                                 keep_default_na=False)
    df_test_vua_allpos = pd.read_csv("../data/VUA/tokenized/test_vua_allpos_tokenized.csv", index_col='token_id',
                                     keep_default_na=False)
    df_test_toefl_allpos = pd.read_csv("../data/TOEFL/tokenized/test_toefl_allpos_tokenized.csv", index_col='token_id',
                                       keep_default_na=False)

    print("VUA Train")
    prepare_inputs(df_train_vua)
    print()
    print("TOEFL Train")
    prepare_inputs(df_train_toefl)
    print()
    print("VUA Test")
    prepare_inputs(df_test_vua_allpos)
    print()
    print("TOEFL Test")
    prepare_inputs(df_test_toefl_allpos)


if __name__ == '__main__':
    main()
