import os
import pickle
import gensim
import pandas as pd
from dotenv import load_dotenv
from collections import defaultdict
from app.ml.processing import TextNormalizer

load_dotenv()
data_path = os.environ.get('DATA_PATH')

# Modelling artifacts
dictionary = gensim.corpora.Dictionary.load(data_path + '/simple_model/dictionary.dict')
tfidf = gensim.models.TfidfModel.load(data_path + '/simple_model/model.tfidf')
index = gensim.similarities.MatrixSimilarity.load(data_path + '/simple_model/queries_index.index')

# Dataset for reference
data = pd.DataFrame(pickle.load(open(data_path + '2020_10_28_22_14_hotel_reviews.pickle', 'rb')))
data.columns = ['name', 'url', 'description', 'amenities', 'reviews', 'rating']

normalizer = TextNormalizer()

def query(query: str, columns: list, n_match: int):
    results_dict = defaultdict(list)
    ids = search_similar(query, n_match)
    
    if 'rating' not in columns:
        columns.append('rating')

    result_df = data.loc[ids, columns]
    result_df = result_df.sort_values('rating', ascending=False)
    
    for col in columns:
        results_dict[col] = result_df.loc[:, col].to_list()

    return results_dict
    

def search_similar(query: str, n_match: int):
    clean_query = normalizer.clean(query)
    bow_query = dictionary.doc2bow(clean_query)
    tfidf_query = tfidf[bow_query]
    sims = index[tfidf_query]

    sorted_ids = []
    similarities = sorted(enumerate(sims), key=lambda item: -item[1])
    for i, sim in similarities[:n_match]:
        sorted_ids.append(i)

    return sorted_ids