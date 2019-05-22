import numpy as np
import tensorflow as tf
from random import randint
import datetime
import os
from pathlib import Path
import pickle as pk
import glob

import implementation as imp

import re ## new import
from collections import Counter

def load_data(path='./data/validate'):
    """
    Load raw reviews from text files, and apply preprocessing
    Append positive reviews first, and negative reviews second
    RETURN: List of strings where each element is a preprocessed review.
    """
    print("Loading IMDB Data...")
    data = []

    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir, path + '/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir, path + '/neg/*')))
    print("Parsing %s files" % len(file_list))
    for i, f in enumerate(file_list):
        with open(f, "r") as openf:
            #print ("\n"+f)
            s = openf.read()
            data.append(preprocess(s))  # NOTE: Preprocessing code called here on all reviews
    return data

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    #print ("original review is:\n",review)
    #print ()
    ## start to preprocess
    #re.sub(pattern, repl, string, count=0, flags=0)¶
    review = review.lower()
    review = re.sub(r"<br />", " ", review)

    review = re.sub(r"won't", "will not ", review)
    review = re.sub(r"won't", "will not ", review)
    review = re.sub(r"don't", "do not ", review)
    review = re.sub(r"didn't", "did not ", review)
    review = re.sub(r"isn't", "be not ", review)
    review = re.sub(r"wasn't", "be not ", review)
    review = re.sub(r"'s", " ", review)
    review = re.sub(r"’s", " ", review)
    review = re.sub(r"'ve", " have ", review)
    review = re.sub(r"'ll", " will ", review)
    review = re.sub(r"'re", " are ", review)
    review = re.sub(r"'d", " ", review)
    review = re.sub(r"i'm", "i am ", review)
    review = re.sub(r"iam", "i am ", review)

    ## movie, film, could, movies
    review = re.sub(r"movie", " ", review)
    review = re.sub(r"film", " ", review)
    review = re.sub(r"could", " ", review)
    review = re.sub(r"movies", " ", review)

    review = re.sub(r"sssssssttttttttttuuuuuuuuuuuuuuuuuupppppiiiddd", "stupid", review)
    review = re.sub(r"goooooooooooooooooooood", "good", review)
    review = re.sub(r"jeeeeeeeesussssssssss", "jesus", review)
    review = re.sub(r"oooops", " ", review)
    review = re.sub(r"ammmmmbbbererrrrrrrrrgerrrrrrrssss", " ", review)
    review = re.sub(r"ennnnnnnnndddddd", "end", review)
    review = re.sub(r"whateverherlastnameis", "what ever her last name is", review)
    review = re.sub(r"annnndddddd", "and", review)
    review = re.sub(r"baaaaaaaaaddddddd", "bad", review)

    review = re.sub('[,.";!?:\(\)-/$\'%`=><“·^\{\}_&#»«\[\]~|@、´，]+', " ", review)
    review = re.sub('[0-9]', " ", review)
    review = re.sub(' [a-z] ', " ", review)
    new_review = [word for word in review.split() if word not in stop_words and len(word) > 2 and word.isalpha()]

    return (" ".join(new_review))

    #return (new_review)

def write_into_temp(filename="tempValidate.txt"):
    review = load_data()
    #review = ["abc sadf", 'bcd asdf', 'def erag', 'asdf ret']
    with open(filename, "w") as f:
        for i in review:
            f.writelines(i+'\n')

write_into_temp()
