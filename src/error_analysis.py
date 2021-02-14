import pandas as pd
import numpy as np
from spellchecker import SpellChecker
from transformers import RobertaTokenizer
from constants import *

vua_verb_errors = "../data/VUA/VUA_Verb_mistakes.csv"
vua_allpos_errors = "../data/VUA/VUA_All_POS_mistakes.csv"
toefl_verb_errors = "../data/TOEFL/TOEFL_Verb_mistakes.csv"
toefl_allpos_errors = "../data/TOEFL/TOEFL_All_POS_mistakes.csv"


def main():
    pd.set_option('display.width', 0)
    pd.set_option('display.max_rows', 100)

    df_vua_verb_errors = pd.read_csv(vua_verb_errors)
    df_vua_allpos_errors = pd.read_csv(vua_allpos_errors)
    df_toefl_verb_errors = pd.read_csv(toefl_verb_errors)
    df_toefl_allpos_errors = pd.read_csv(toefl_allpos_errors)

    all_dfs = {'VUA Verb': df_vua_verb_errors,
               'VUA All POS': df_vua_allpos_errors,
               'TOEFL Verb': df_toefl_verb_errors,
               'TOEFL All POS': df_toefl_allpos_errors}

    spell = SpellChecker()

    for name, df in all_dfs.items():
        print(name)

        print(f'Metaphors: {int(np.sum(df["metaphor"]))}, Literals: {int(len(df) - np.sum(df["metaphor"]))}, Total: '
              f'{len(df)}')
        misspellings = spell.unknown(df['query'])
        print(f'Number of misspellings: {len(misspellings)} out of {len(df)}')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        global_context_input = df['sentence'] + " </s> " + df['query'] + " </s> " + df['pos'] + " </s> " + df['fgpos']
        local_context_input = df['local'] + " </s> " + df['query'] + " </s> " + df['pos'] + " </s> " + df['fgpos']
        global_tokenized_lens = [len(tokenizer(inp)) > max_seq_len for inp in global_context_input]
        local_tokenized_lens = [len(tokenizer(inp)) > max_seq_len for inp in local_context_input]
        print(f'Number of tokenized sentences longer than max_seq_len: {sum(global_tokenized_lens)}')
        print(f'Number of tokenized local contexts longer than max_seq_len: {sum(local_tokenized_lens)}')
        print()
        print(df['pos'].value_counts())
        print()
        print(df['fgpos'].value_counts())
        print()
        print(df.head())
        print()
        print()


if __name__ == '__main__':
    main()
