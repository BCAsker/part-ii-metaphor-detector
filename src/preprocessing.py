import numpy as np
import pandas as pd
import re
import spacy
import os
from spacy.tokenizer import Tokenizer
from spellchecker import SpellChecker

vua_train = "../data/VUA/vuamc_corpus_train.csv"
vua_test = "../data/VUA/vuamc_corpus_test.csv"
vua_train_gold = "../data/VUA/naacl_flp_train_gold_labels/all_pos_tokens.csv"
vua_allpos_test_gold = "../data/VUA/naacl_flp_test_gold_labels/all_pos_tokens.csv"
vua_verb_test_gold = "../data/VUA/naacl_flp_test_gold_labels/verb_tokens.csv"

toefl_train = "../data/TOEFL/toefl_sharedtask_dataset/essays"
toefl_test = "../data/TOEFL/toefl_sharedtask_evaluation_kit/essays_with_labels"
toefl_train_required = "../data/TOEFL/toefl_skll_train_features/features/all_pos/P.jsonlines"
toefl_allpos_test_required = "../data/TOEFL/toefl_sharedtask_evaluation_kit/toefl_all_pos_test_tokens.csv"
toefl_verb_test_required = "../data/TOEFL/toefl_sharedtask_evaluation_kit/toefl_verb_test_tokens.csv"

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

    spell = SpellChecker()
    originals = [re.sub('M_', '', sentence) for sentence in df_raw['sentence_txt']]
    corrected = [' '.join([spell.correction(word) for word in sentence.split()]) for sentence in originals]
    docs = nlp.pipe(corrected, disable=['ner'])

    for doc, text_id, sentence_id in zip(docs, df_raw['txt_id'], df_raw['sentence_id']):
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

    # Get the tokens with the correct POS, etc. for training from the given file
    df_required = pd.read_csv(vua_train_gold, encoding='windows-1252', header=None, names=['metaphor']).dropna()
    df_required.index.names = ['token_id']
    df_out = df_out.loc[df_required.index]

    return df_out


# Create (S, qi, yj) triples from the VUA data, as defined in DeepMet paper. Use spacy to tokenize and get POS tags
# One row for each token
def create_vua_test_dataframe(is_verb_task=False):
    df_raw = pd.read_csv(vua_test, encoding='windows-1252').dropna()

    tokens = []
    token_ids = []
    course_pos_tags = []
    fine_pos_tags = []
    sentences = []
    local_contexts = []

    spell = SpellChecker()
    corrected = [' '.join([spell.correction(word) for word in sentence.split()]) for sentence in df_raw['sentence_txt']]
    docs = nlp.pipe(corrected, disable=['ner'])

    for doc, text_id, sentence_id in zip(docs, df_raw['txt_id'], df_raw['sentence_id']):
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

    required_file = vua_verb_test_gold if is_verb_task else vua_allpos_test_gold
    df_metaphor = pd.read_csv(required_file, encoding='windows-1252', header=None, names=['metaphor']).dropna()

    df_out = df_out.join(df_metaphor)
    df_out = df_out[df_out['metaphor'].notna()]
    df_out.astype({'metaphor': 'int64'})

    return df_out[['sentence', 'local', 'query', 'metaphor', 'pos', 'fgpos']]


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

                spell = SpellChecker()
                originals = [re.sub('M_', '', sentence.strip()) for sentence in file]
                corrected = [' '.join([spell.correction(word) for word in sentence.split()]) for sentence in originals]
                docs = nlp.pipe(corrected, disable=['ner'])

                for doc in docs:
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

    df_required = pd.read_json(toefl_train_required, lines=True)
    required_ids = ['_'.join((token_id.split('_')[:-1])) for token_id in df_required['id']]
    df_out = df_out.loc[required_ids]

    return df_out


# Create (S, qi, yj) triples from the TOEFL data, as defined in DeepMet paper. Use spacy to tokenize and get POS tags
# One row for each token
def create_toefl_test_dataframe(is_verb_task=False):
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

                spell = SpellChecker()
                corrected = [' '.join([spell.correction(word) for word in sentence.strip().split()])
                             for sentence in file]
                docs = nlp.pipe(corrected, disable=['ner'])

                for doc in docs:
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

    required_file = toefl_verb_test_required if is_verb_task else toefl_allpos_test_required
    df_required = pd.read_csv(required_file, header=None, names=['id']).dropna()
    required_ids = ['_'.join((token_id.split('_')[:-1])) for token_id in df_required['id']]
    df_out = df_out.loc[required_ids]

    return df_out


def check_preprocessed_files_saved():
    return os.path.exists("../data/VUA/tokenized/train_vua_tokenized.csv") and \
        os.path.exists("../data/VUA/tokenized/test_vua_verb_tokenized.csv") and \
        os.path.exists("../data/VUA/tokenized/test_vua_allpos_tokenized.csv") and \
        os.path.exists("../data/TOEFL/tokenized/train_toefl_tokenized.csv") and \
        os.path.exists("../data/TOEFL/tokenized/test_toefl_verb_tokenized.csv") and \
        os.path.exists("../data/TOEFL/tokenized/test_toefl_allpos_tokenized.csv")


# Preprocess data into (S, qi, yj) triples
def initial_preprocessing():
    df_vua_train = create_vua_train_dataframe()
    df_vua_test_verb = create_vua_test_dataframe(is_verb_task=True)
    df_vua_test_allpos = create_vua_test_dataframe(is_verb_task=False)

    df_toefl_train = create_toefl_train_dataframe()
    df_toefl_test_verb = create_toefl_test_dataframe(is_verb_task=True)
    df_toefl_test_allpos = create_toefl_test_dataframe(is_verb_task=False)

    df_vua_train.to_csv("../data/VUA/tokenized/train_vua_tokenized.csv")
    df_vua_test_verb.to_csv("../data/VUA/tokenized/test_vua_verb_tokenized.csv")
    df_vua_test_allpos.to_csv("../data/VUA/tokenized/test_vua_allpos_tokenized.csv")
    df_toefl_train.to_csv("../data/TOEFL/tokenized/train_toefl_tokenized.csv")
    df_toefl_test_verb.to_csv("../data/TOEFL/tokenized/test_toefl_verb_tokenized.csv")
    df_toefl_test_allpos.to_csv("../data/TOEFL/tokenized/test_toefl_allpos_tokenized.csv")


if __name__ == '__main__':
    initial_preprocessing()
