# https://socialnetwork.readthedocs.io/en/latest/doc_similarity.html

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    
    # Your Code Here
    # ref) https://socialnetwork.readthedocs.io/en/latest/doc_similarity.html
    token = nltk.word_tokenize(doc)
    tag = nltk.pos_tag(token) # add parts of speech to token
    nltk2wordnet = [(w, convert_tag(t)) for (w,t) in tag] # convert nltk pos into wordnet pos
    output = [wn.synsets(w,t)[0] for (w, t) in nltk2wordnet if len(wn.synsets(w,t)) > 0]
    return output


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    
    # Your Code Here
    list1 = []
    # For each synset in s1
    for a in s1:
        # finds the synset in s2 with the largest similarity value
        list1.append(max([i.path_similarity(a) for i in s2 if i.path_similarity(a) is not None]))
    output = sum(list1)/len(list1)
    return output


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2

###########################################################################333
def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets \
    and similarity_score is correct!'
    return document_path_similarity(doc1, doc2)

test_document_path_similarity() # 0.5426445578231293

# Use this dataframe for questions most_similar_docs and label_accuracy
paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head(10)

def most_similar_docs():
    
    # Your Code Here
    def func(x):
        try:
            return document_path_similarity(str(x['D1']), str(x['D2']))
        except:
            return np.nan
    paraphrases['similarity_score'] = paraphrases.apply(func, axis = 1)
    
    # sort by score and extract the max
    df = paraphrases.sort_values('similarity_score', ascending=False)[:1]
    # remove similarity score
    df = df[df.columns[1:]]
    # change dataframe to an array, and convert to a tuple
    output = tuple(df.values[0])

    return output

print(most_similar_docs())

"""
Provide labels for the twenty pairs of documents by computing the similarity for each pair using `document_path_similarity`. 
Let the classifier rule be that if the score is greater than 0.75, label is paraphrase (1), else label is not paraphrase (0). 
Report accuracy of the classifier using scikit-learn's accuracy_score.
"""
def label_accuracy():
    from sklearn.metrics import accuracy_score

    # Your Code Here
    def func(x):
        try:
            return document_path_similarity(x['D1'], x['D2'])
        except:
            return np.nan

    paraphrases['similarity_score'] = paraphrases.apply(func, axis=1)
    df = paraphrases
    df2 = df.dropna()
#     df2['label'] = df2['similarity_score'].apply(lambda x: 1 if x > 0.75 else 0)
    df2['label'] = df2['similarity_score'].apply(lambda x: 1 if x > 0.70 else 0)


    output = accuracy_score(df2['label'], df2['Quality'])

    return output

print(label_accuracy())
paraphrases
