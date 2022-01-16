# -*- coding: utf-8 -*-
import click
import torch
from dotenv import find_dotenv, load_dotenv
from spacy.lang.en import English
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
from csv import reader
import pandas as pd
import sys
import glob
import os
import re
import time
import csv
import spacy
import html
import numpy as np

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe("sentencizer")

FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}
PUNCTS = {'#', '$', '.', ',', ':', '(', ')', '"', "``", "`", "'", "\'\'"}

# Path of lexical norm files and index of columns to extract
bgl_path = r"src/data/Wordlists/BristolNorms+GilhoolyLogie.csv"
war_path = r"src/data/Wordlists/Ratings_Warriner_et_al.csv"
bgl_col = [1,3,4,5]
war_col = [1,2,5,8]

# Mapping between word tag and feature array index
tag_idx_dict = {"CC":4, "VBD":5, "NN":9, "NNS": 9, "NNP":10, "NNPS":10,
                "RB":11, "RBR":11, "RBS":11, "WDT":12, "WP":12, "WP$":12, "WRB":12}

# class TweetDataset(Dataset):
#     def __init__(self, inputfile_path, )

class TweetDataset(Dataset):
    def __init__(self, input_filepath):
        # for filename in os.listdir(input_filepath):
        os.chdir(input_filepath)
        extension = 'csv'
        all_csv = [i for i in glob.glob('*.{}'.format(extension))]
        df = pd.concat([pd.read_csv(f) for f in all_csv])
        df['label (depression result)'] = df['label (depression result)'].astype(int)
        df['features'] = df['message to examine'].apply(process_tweet)

        labels_tensor  = torch.tensor(df['label (depression result)'].values)
        features_tensor = torch.cat(df['features'].tolist(), dim = 0)

        self.labels = labels_tensor
        self.features = features_tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return self.features[idx, :], self.labels[idx]

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    global bgl_dict, bgl_df, war_dict, war_df
    bgl_dict, bgl_df = read_word_score(bgl_path, bgl_col)
    war_dict, war_df = read_word_score(war_path, war_col)

    dataset = TweetDataset(input_filepath)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    ## logger = logging.getLogger(__name__)
    ## logger.info('making final data set from raw data')

    # for filename in os.listdir(input_filepath):
    #     if filename.endswith('.csv'):
    #         filepath = os.path.join(input_filepath, filename)
    #         df = pd.read_csv(filepath)
    #         df['label (depression result)'] = df['label (depression result)'].astype(int)
    #         df['preprocessed'] = df['message to examine'].apply(preproc)
    #         df['features'] = df['preprocessed'].apply(extract_features)


    # features_tensor = torch.cat(df['features'].tolist(), dim = 0)
    # labels_tensor  = torch.tensor(df['label (depression result)'].values)

    # labels_out_path = os.path.join(output_filepath, 'labels')
    # torch.save(labels_tensor, labels_out_path)

    # features_out_path = os.path.join(output_filepath, 'features')
    # torch.save(features_tensor, features_out_path)

    

    return None

def process_tweet(tweet):
    preprocssed = preproc(tweet)
    features = extract_features(preprocssed)
    return features


def preproc(tweet, steps=range(1, 6)):
    ''' This function pre-processes a single tweet
    Parameters:
        tweet : string, the body of a tweet
        steps   : list of ints, each entry in this list corresponds to a preprocessing step
    Returns:
        modified_tweet : string, the modified tweet
    '''
    modified_tweet = tweet
    if 1 in steps:
        # replace newlines, tabs and carriage returns with spaces
        modified_tweet = re.sub(r"[\n\t\r]{1,}", " ", modified_tweet)

    if 2 in steps:
        # unescape html and replace html character codes with ascii equivalent
        modified_tweet= html.unescape(modified_tweet)

    if 3 in steps:
        # remove URLs
        modified_tweet = re.sub(r"(http|www)\S+", "", modified_tweet)

    if 4 in steps:
        # remove duplicate spaces.
        modified_tweet = re.sub(' +', ' ', modified_tweet)

    if 5 in steps:
        doc = nlp(modified_tweet)

        modified_tweet = ""
        for sentence in doc.sents:
            for token in sentence:
                # Case for pronouns such as "I"
                if (not token.text.startswith("-")) and token.lemma_.startswith("-"):
                    content = token.text
                else:
                    content = token.lemma_

                # Case when the text is entirely written in upper case or not
                if token.text.isupper():
                    modified_tweet += content.upper()
                else:
                    modified_tweet += content.lower()

                # Add Tag & Space after each token
                modified_tweet = modified_tweet + "/" + token.tag_ + " "

            # Insert "\n" between sentences
            modified_tweet = modified_tweet[:-1] + "\n"

    return modified_tweet

def read_word_score(path, col_lst):
    """
    A helper function to read lexical norm files
    :param path: str, path of the file
    :param col_lst: list[int], index of columns in the csv files that needs to be extracted
    :return: tuple[dict, np.ndarray], returning a dictionary of <word, index> pairs and an array that stores the scores
    """
    # Initialize Nested List to store word and its scores
    word_df = [[] for _ in range(len(col_lst))]

    # Read the file
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader, None)
        for line in reader:
            data = ','.join(line).split(',')
            for word_df_idx, col_idx in enumerate(col_lst):
                if data[col_idx] == '':
                    break
                word_df[word_df_idx].append(data[col_idx])

    # Create a dictionary that maps from word to an arbitray index
    word_dict = dict(zip(word_df[0], [x for x in range(len(word_df[0]))]))

    # Create an array that stores the scores only, and convert it to float.
    word_df = np.array(word_df[1:])
    word_df = word_df.astype(np.float64)

    return word_dict, word_df

def extract_features(tweet):
    ''' This function extracts features from a single tweet
    Parameters:
        tweet : string, the body of a comment (after preprocessing)
    Returns:
        features : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''

    # Initialize Feature Array
    feature_arr = np.zeros(29)

    # Initialize sentence count, token count, character count, and token (excluding punctuation count)
    sent_cnt = 0
    token_cnt = 0
    char_cnt = 0
    token_ex_punct_cnt = 0

    # Initialize two lists to track words that appear in the lexical norm data
    war_lemma_lst = []
    bgl_lemma_lst = []


    # Set up to deal with fixed multiword vocabs such as "United States", "United Arab Emirates" and etc.
    phrase = False
    phrase_word = ''

    # Iterate over tokens that are separated by space
    for token in tweet.split():
        if phrase:
            token = phrase_word + token
            phrase_word = ''
            phrase = False
        # Handling special cases where the token = ' ' or ''
        if "/" not in token:
            phrase = True
            phrase_word += token + " "
            continue
        else:
            # Split the token into lemma and tag
            lemma, tag = token[:token.rindex("/")], token[token.rindex("/") + 1:]

            # Avoid Empty String
            if len(lemma) < 1:
                continue

            # Feature 1
            if len(lemma) >= 3 and lemma.isupper():
                feature_arr[0] += 1

            # Use lowercase lemma
            lemma = lemma.lower()

            # Feature that depends on lemma value only (2,3,4,8,14)
            if lemma in FIRST_PERSON_PRONOUNS:
                feature_arr[1] += 1
            elif lemma in SECOND_PERSON_PRONOUNS:
                feature_arr[2] += 1
            elif lemma in THIRD_PERSON_PRONOUNS:
                feature_arr[3] += 1
            elif lemma in SLANG:
                feature_arr[13] += 1
            elif lemma == ",":
                feature_arr[7] += 1

            # Feature that depends on tag value only (5,6, 10, 11, 12, 13)
            if tag in tag_idx_dict:
                feature_arr[tag_idx_dict[tag]] += 1

            # Feature that depends on tag and lemma (7, 9)
            # Case of Future Tense that only depends on 1 tag
            if (tag == "MD" and lemma == "will"):
                feature_arr[6] += 1
            elif tag in PUNCTS and len(lemma) > 1:
                feature_arr[8] += 1

            # Track words that appear in lexical norm dictionary, and do character/token count
            if tag not in PUNCTS:
                char_cnt += len(lemma)
                token_ex_punct_cnt += 1

                if lemma in war_dict:
                    war_lemma_lst.append(war_dict[lemma])
                if lemma in bgl_dict:
                    bgl_lemma_lst.append(bgl_dict[lemma])

            token_cnt += 1

    # Case of Future Tense that in the format "go to do"
    feature_arr[6] += len([*re.finditer("(go|GO)/VBG (to|TO)/TO .*(/VB)", tweet)])

    # Add sentence count. Note if a sentence does not have a \n, it defaults to 1 sentence if the comment is not an empty string
    sent_cnt = max([len([*re.finditer("\n", tweet)]), 1]) if tweet != '' else 0

    # Avg Length of Sentences
    if sent_cnt > 0:
        feature_arr[14] += (token_cnt/sent_cnt)
    # Avg Length of Token
    if token_ex_punct_cnt > 0:
        feature_arr[15] += (char_cnt/token_ex_punct_cnt)
    # Number of sentences
    feature_arr[16] += sent_cnt

    # Lexical norm statistics (mean & stdev)
    if len(bgl_lemma_lst) > 0:
        feature_arr[17:20] = np.mean(bgl_df[:, bgl_lemma_lst], axis=1)
        feature_arr[20:23] = np.std(bgl_df[:, bgl_lemma_lst], axis=1)

    if len(war_lemma_lst) > 0:
        feature_arr[23:26] = np.mean(war_df[:, war_lemma_lst], axis=1)
        feature_arr[26:29] = np.std(war_df[:, war_lemma_lst], axis=1)

    # If any the statistical value is not applicable, set to default = 0
    # feature_arr[np.isnan(feature_arr)] = 0
    feature_arr = np.nan_to_num(feature_arr) 
    feature_arr = feature_arr.astype(np.float64)
    return torch.tensor(feature_arr).resize_(1, 29)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()