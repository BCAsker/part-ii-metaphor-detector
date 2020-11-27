import numpy as np
import pandas as pd
import re
import spacy
import os
from spacy.tokenizer import Tokenizer

vua_train = "../data/VUA/vuamc_corpus_train.csv"
vua_train_gold = "../data/VUA/naacl_flp_train_gold_labels/all_pos_tokens.csv"

toefl_train = "../data/TOEFL/toefl_sharedtask_dataset/essays"

nlp = spacy.load("en_core_web_sm")
# Use a blank tokenizer. Provides the same tokenization as string.split(), which is the format used for the shared task
nlp.tokenizer = Tokenizer(nlp.vocab)


# Checks the leading two characters of a token to see if it's a metaphor
def is_metaphor(token):
    metaphor = 0
    if len(token) >= 2:
        if token[0:2] == 'M_':
            metaphor = 1
    return metaphor


# Create (S, qi, yj) triples from the VUA data, as defined in DeepMet paper. Use spacy to tokenize and get POS tags
def create_vua_dataframe():
    df_raw = pd.read_csv(vua_train, encoding='windows-1252').dropna()

    metaphors = []
    for split in [sentence.split() for sentence in df_raw['sentence_txt']]:
        metaphors.extend([is_metaphor(token) for token in split])

    tokens = []
    token_ids = []
    course_tags = []
    fine_tags = []
    sentences = []
    for doc, text_id, sentence_id in zip(
            nlp.pipe([re.sub('M_', '', sentence) for sentence in df_raw['sentence_txt']], disable=['ner']),
            df_raw['txt_id'], df_raw['sentence_id']):

        for offset, token in zip(range(1, len(doc) + 1), doc):
            tokens.append(token.text)
            token_ids.append(text_id + '_' + sentence_id + '_' + str(offset))
            course_tags.append(token.pos)
            fine_tags.append(token.tag)
            sentences.append(str(doc))

    df_out = pd.DataFrame(np.array([sentences, tokens, metaphors, course_tags, fine_tags]).T,
                          columns=['sentence', 'query', 'metaphor', 'pos', 'fgpos'],
                          index=token_ids)

    df_out.index.names = ['token_id']
    df_out[['metaphor', 'pos', 'fgpos']] = df_out[['metaphor', 'pos', 'fgpos']].apply(pd.to_numeric)

    return df_out


# Create (S, qi, yj) triples from the TOEFL data, as defined in DeepMet paper. Use spacy to tokenize and get POS tags
def create_toefl_dataframe():
    tokens = []
    metaphors = []
    token_ids = []
    course_tags = []
    fine_tags = []
    sentences = []

    for (root, _, filenames) in os.walk(toefl_train):
        for filename in filenames:
            with open(root + '/' + filename, 'r') as file:
                metaphors.extend([is_metaphor(token) for sentence in file for token in sentence.split()])
                file.seek(0)
                sentence_id = 1
                for doc in nlp.pipe([re.sub('M_', '', sentence.strip()) for sentence in file], disable=['ner']):
                    for offset, token in zip(range(1, len(doc) + 1), doc):
                        tokens.append(token.text)
                        token_ids.append(filename[:-4] + '_' + str(sentence_id) + '_' + str(offset))
                        course_tags.append(token.pos)
                        fine_tags.append(token.tag)
                        sentences.append(str(doc))
                    sentence_id += 1

    df_out = pd.DataFrame(np.array([sentences, tokens, metaphors, course_tags, fine_tags]).T,
                          columns=['sentence', 'query', 'metaphor', 'pos', 'fgpos'],
                          index=token_ids)
    df_out.index.names = ['token_id']
    df_out[['metaphor', 'pos', 'fgpos']] = df_out[['metaphor', 'pos', 'fgpos']].apply(pd.to_numeric)

    return df_out


# Preprocess data into (S, qi, yj) triples
def main():
    df_vua_train = create_vua_dataframe()
    df_toefl_train = create_toefl_dataframe()

    # Make sure we've read the VUA dataset correctly by comparing to the gold labels
    df_vua_train_gold = pd.read_csv(vua_train_gold, header=None, names=['token_id', 'metaphor'], index_col='token_id')
    df_vua_train_gold.sort_index(inplace=True)
    vua_labels_from_text = df_vua_train.loc[df_vua_train_gold.index, 'metaphor']
    assert np.all(vua_labels_from_text.values == df_vua_train_gold['metaphor'])

    df_vua_train.to_csv("../data/VUA/vua_train_tokenized.csv")
    df_toefl_train.to_csv("../data/TOEFL/toefl_train_tokenized.csv")


if __name__ == '__main__':
    main()
