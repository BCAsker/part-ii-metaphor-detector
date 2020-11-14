import numpy as np
import pandas as pd
import re
import spacy
from spacy.tokenizer import Tokenizer

vua_train = "../data/VUA/vuamc_corpus_train.csv"
vua_train_gold = "../data/VUA/naacl_flp_train_gold_labels/all_pos_tokens.csv"

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


# Create (S, qi, yj) triples, as defined in DeepMet paper. Use spacy to tokenize and get POS tags
def create_tokenized_dataframe(df_raw):
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


# Preprocess data into (S, qi, yj) triples
def main():
    df_train = pd.read_csv(vua_train, encoding='windows-1252').dropna()
    df_train_tokenized = create_tokenized_dataframe(df_train)
    df_train_gold = pd.read_csv(vua_train_gold, header=None, names=['token_id', 'metaphor'], index_col='token_id')

    df_train_gold.sort_index(inplace=True)
    labels_from_text = df_train_tokenized.loc[df_train_gold.index, 'metaphor']
    assert np.all(labels_from_text.values == df_train_gold['metaphor'].values)

    df_train_tokenized.to_csv("../data/VUA/vua_train_tokenized.csv")


if __name__ == '__main__':
    main()
