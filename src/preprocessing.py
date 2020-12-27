import numpy as np
import pandas as pd
import re
import spacy
import os
from spacy.tokenizer import Tokenizer

vua_train = "../data/VUA/vuamc_corpus_train.csv"
vua_test = "../data/VUA/vuamc_corpus_test.csv"
vua_train_gold = "../data/VUA/naacl_flp_train_gold_labels/all_pos_tokens.csv"
vua_test_gold = "../data/VUA/naacl_flp_test_gold_labels/all_pos_tokens.csv"

toefl_train = "../data/TOEFL/toefl_sharedtask_dataset/essays"
toefl_test = "../data/TOEFL/toefl_sharedtask_evaluation_kit/essays_with_labels"

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
# One row for each token
def create_vua_train_dataframe():
    df_raw = pd.read_csv(vua_train, encoding='windows-1252').dropna()

    metaphors = []
    for split in [sentence.split() for sentence in df_raw['sentence_txt']]:
        metaphors.extend([is_metaphor(token) for token in split])

    tokens = []
    token_ids = []
    course_pos_tags = []
    fine_pos_tags = []
    sentences = []
    local_contexts = []

    for doc, text_id, sentence_id in zip(
            nlp.pipe([re.sub('M_', '', sentence) for sentence in df_raw['sentence_txt']], disable=['ner']),
            df_raw['txt_id'], df_raw['sentence_id']):

        split_sentence = str(doc).split(', ')
        split_index = 0

        for offset, token in zip(range(1, len(doc) + 1), doc):
            tokens.append(token.text)
            token_ids.append(text_id + '_' + sentence_id + '_' + str(offset))
            course_pos_tags.append(token.pos_)
            fine_pos_tags.append(token.tag_)
            sentences.append(str(doc))
            local_contexts.append(split_sentence[split_index])
            if token.text == ',':
                split_index += 1

    df_out = pd.DataFrame(np.array([sentences, local_contexts, tokens, metaphors, course_pos_tags, fine_pos_tags]).T,
                          columns=['sentence', 'local', 'query', 'metaphor', 'pos', 'fgpos'],
                          index=token_ids)
    df_out.index.names = ['token_id']

    return df_out


# Create (S, qi, yj) triples from the VUA data, as defined in DeepMet paper. Use spacy to tokenize and get POS tags
# One row for each token
def create_vua_test_dataframe():
    df_raw = pd.read_csv(vua_test, encoding='windows-1252').dropna()

    tokens = []
    token_ids = []
    course_pos_tags = []
    fine_pos_tags = []
    sentences = []
    local_contexts = []

    for doc, text_id, sentence_id in zip(nlp.pipe(df_raw['sentence_txt'], disable=['ner']), df_raw['txt_id'],
                                         df_raw['sentence_id']):

        split_sentence = str(doc).split(', ')
        split_index = 0

        for offset, token in zip(range(1, len(doc) + 1), doc):
            tokens.append(token.text)
            token_ids.append(str(text_id) + '_' + str(sentence_id) + '_' + str(offset))
            course_pos_tags.append(token.pos_)
            fine_pos_tags.append(token.tag_)
            sentences.append(str(doc))
            local_contexts.append(split_sentence[split_index])
            if token.text == ',':
                split_index += 1

    df_out = pd.DataFrame(np.array([sentences, local_contexts, tokens, course_pos_tags, fine_pos_tags]).T,
                          columns=['sentence', 'local', 'query', 'pos', 'fgpos'],
                          index=token_ids)
    df_out.index.names = ['token_id']

    df_metaphor = pd.read_csv(vua_test_gold, encoding='windows-1252', header=None, names=['metaphor']).dropna()

    df_out = df_out.join(df_metaphor)
    df_out = df_out[df_out['metaphor'].notna()]
    df_out.astype({'metaphor': 'int64'})

    return df_out[['sentence', 'local', 'query', 'metaphor', 'pos', 'fgpos']]


# Create (S, qi, yj) triples from the VUA data, as defined in DeepMet paper. Use spacy to tokenize and get POS tags
# One row for each sentence
def create_vua_dataframe_compact():
    df_raw = pd.read_csv(vua_train, encoding='windows-1252').dropna()

    metaphors = []
    sentences = []
    for index, row in df_raw.iterrows():
        metaphors.append([is_metaphor(token) for token in row['sentence_txt'].split()])
        sentences.append(re.sub('M_', '', row['sentence_txt']))

    tokens = []
    coarse_pos_tags = []
    fine_pos_tags = []
    for doc in nlp.pipe(sentences, disable=['ner']):
        tokens.append(list(doc))
        coarse_pos_tags.append([doc.vocab.strings[pos] for pos in doc.to_array('pos')])
        fine_pos_tags.append([doc.vocab.strings[tag] for tag in doc.to_array('tag')])

    sentence_ids = [txt_id + '_' + sentence_id for txt_id, sentence_id in zip(df_raw['txt_id'], df_raw['sentence_id'])]

    df_out = pd.DataFrame(np.array([sentences, tokens, metaphors, coarse_pos_tags, fine_pos_tags], dtype=object).T,
                          columns=['sentence', 'tokens', 'metaphors', 'pos', 'fgpos'],
                          index=sentence_ids)
    df_out.index.names = ['sentence_id']

    return df_out


# Create (S, qi, yj) triples from the TOEFL data, as defined in DeepMet paper. Use spacy to tokenize and get POS tags
# One row for each token
def create_toefl_train_dataframe():
    tokens = []
    metaphors = []
    token_ids = []
    course_pos_tags = []
    fine_pos_tags = []
    sentences = []
    local_contexts = []

    for (root, _, filenames) in os.walk(toefl_train):
        for filename in filenames:
            with open(root + '/' + filename, 'r') as file:
                metaphors.extend([is_metaphor(token) for sentence in file for token in sentence.split()])
                file.seek(0)
                sentence_id = 1
                for doc in nlp.pipe([re.sub('M_', '', sentence.strip()) for sentence in file], disable=['ner']):
                    split_sentence = str(doc).split(', ')
                    split_index = 0
                    for offset, token in zip(range(1, len(doc) + 1), doc):
                        tokens.append(token.text)
                        token_ids.append(filename[:-4] + '_' + str(sentence_id) + '_' + str(offset))
                        course_pos_tags.append(token.pos_)
                        fine_pos_tags.append(token.tag_)
                        sentences.append(str(doc))
                        local_contexts.append(split_sentence[split_index])
                        if token.text == ',':
                            split_index += 1
                    sentence_id += 1

    df_out = pd.DataFrame(np.array([sentences, local_contexts, tokens, metaphors, course_pos_tags, fine_pos_tags]).T,
                          columns=['sentence', 'local', 'query', 'metaphor', 'pos', 'fgpos'],
                          index=token_ids)
    df_out.index.names = ['token_id']

    return df_out


# Create (S, qi, yj) triples from the TOEFL data, as defined in DeepMet paper. Use spacy to tokenize and get POS tags
# One row for each token
def create_toefl_test_dataframe():
    tokens = []
    metaphors = []
    token_ids = []
    course_pos_tags = []
    fine_pos_tags = []
    sentences = []
    local_contexts = []

    for (root, _, filenames) in os.walk(toefl_test):
        for filename in filenames:
            with open(root + '/' + filename, 'r') as file:
                metaphors.extend([is_metaphor(token) for sentence in file for token in sentence.split()])
                file.seek(0)
                sentence_id = 1
                for doc in nlp.pipe([re.sub('M_', '', sentence.strip()) for sentence in file], disable=['ner']):
                    split_sentence = str(doc).split(', ')
                    split_index = 0
                    for offset, token in zip(range(1, len(doc) + 1), doc):
                        tokens.append(token.text)
                        token_ids.append(filename[:-4] + '_' + str(sentence_id) + '_' + str(offset))
                        course_pos_tags.append(token.pos_)
                        fine_pos_tags.append(token.tag_)
                        sentences.append(str(doc))
                        local_contexts.append(split_sentence[split_index])
                        if token.text == ',':
                            split_index += 1
                    sentence_id += 1

    df_out = pd.DataFrame(np.array([sentences, local_contexts, tokens, metaphors, course_pos_tags, fine_pos_tags]).T,
                          columns=['sentence', 'local', 'query', 'metaphor', 'pos', 'fgpos'],
                          index=token_ids)
    df_out.index.names = ['token_id']

    return df_out


# Create (S, qi, yj) triples from the TOEFL data, as defined in DeepMet paper. Use spacy to tokenize and get POS tags
# One row for each sentence
def create_toefl_dataframe_compact():
    metaphors = []
    sentences = []
    sentence_ids = []
    tokens = []
    coarse_pos_tags = []
    fine_pos_tags = []

    for (root, _, filenames) in os.walk(toefl_train):
        for filename in filenames:
            with open(root + '/' + filename, 'r') as file:
                sentence_num = 1
                for row in file:
                    metaphors.append([is_metaphor(token) for token in row.strip().split()])
                    sentences.append(re.sub('M_', '', row.strip()))
                    sentence_ids.append(filename[:-4] + '_' + str(sentence_num))
                    sentence_num += 1
                file.seek(0)
                for doc in nlp.pipe([re.sub('M_', '', sentence.strip()) for sentence in file], disable=['ner']):
                    tokens.append(list(doc))
                    coarse_pos_tags.append([doc.vocab.strings[pos] for pos in doc.to_array('pos')])
                    fine_pos_tags.append([doc.vocab.strings[tag] for tag in doc.to_array('tag')])

    df_out = pd.DataFrame(np.array([sentences, tokens, metaphors, coarse_pos_tags, fine_pos_tags], dtype=object).T,
                          columns=['sentence', 'tokens', 'metaphors', 'pos', 'fgpos'],
                          index=sentence_ids)
    df_out.index.names = ['sentence_id']

    return df_out


# Preprocess data into (S, qi, yj) triples
def main():
    df_vua_train = create_vua_train_dataframe()
    df_vua_test = create_vua_test_dataframe()
    df_toefl_train = create_toefl_train_dataframe()
    df_toefl_test = create_toefl_test_dataframe()

    # Make sure we've read the VUA dataset correctly by comparing to the gold labels
    df_vua_train_gold = pd.read_csv(vua_train_gold, header=None, names=['token_id', 'metaphor'], index_col='token_id')
    df_vua_train_gold.sort_index(inplace=True)
    vua_labels_from_text = df_vua_train.loc[df_vua_train_gold.index, 'metaphor'].apply(pd.to_numeric)
    assert np.all(vua_labels_from_text.values == df_vua_train_gold['metaphor'])

    df_vua_train.to_csv("../data/VUA/vua_train_tokenized.csv")
    df_vua_test.to_csv("../data/VUA/vua_test_tokenized.csv")
    df_toefl_train.to_csv("../data/TOEFL/toefl_train_tokenized.csv")
    df_toefl_test.to_csv("../data/TOEFL/toefl_test_tokenized.csv")


if __name__ == '__main__':
    main()
