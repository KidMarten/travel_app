import os
import pickle
import gensim
import pandas as pd
from dotenv import load_dotenv
from collections import defaultdict
from ml.processing import TextNormalizer

load_dotenv()
data_path = os.environ.get('DATA_PATH')

# Modelling artifacts
dictionary = gensim.corpora.Dictionary.load(data_path + '/simple_model/dictionary.dict')
tfidf = gensim.models.TfidfModel.load(data_path + '/simple_model/model.tfidf')
lda = gensim.models.LdaModel.load(data_path + '/topic_model/model.lda')

tfidf_index = gensim.similarities.MatrixSimilarity.load(data_path + '/simple_model/queries_index.index')
lda_index = gensim.similarities.MatrixSimilarity.load(data_path + '/topic_model/full_description.index')

# Dataset for reference
data = pd.DataFrame(pickle.load(open(data_path + '/simple_model/2020_10_28_22_14_hotel_reviews.pickle', 'rb')))
data.columns = ['name', 'url', 'description', 'amenities', 'reviews', 'rating']

normalizer = TextNormalizer()


def search_similar(query: str, n_match: int, model:str):
    
    clean_query = normalizer.clean(query)
    bow_query = dictionary.doc2bow(clean_query)

    if model == 'LDA':
        lda_query = lda[bow_query]
        sims = lda_index[lda_query]

    elif model == 'Tf-Idf':
        tfidf_query = tfidf[bow_query]
        sims = tfidf_index[tfidf_query]

    sorted_ids = []
    similarities = sorted(enumerate(sims), key=lambda item: -item[1])
    
    for i, sim in similarities[:n_match]:
        sorted_ids.append(i)

    return sorted_ids


def query(query: str, n_match: int, model: str):

    ids = search_similar(query, n_match, model)

    result_df = data.loc[ids, ['name', 'url', 'rating']]
    result_df.rename(
        columns={'name': 'Name', 'url': 'URL', 'rating': 'Rating'},
        inplace=True
    )
    result_df.reset_index(
        inplace=True,
        drop=True
    )
    result_df.index = result_df.index + 1
    return result_df

