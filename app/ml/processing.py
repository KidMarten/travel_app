from pymystem3 import Mystem
from gensim.utils import simple_preprocess


class TextNormalizer():
    '''Cleanes the string before all transformations'''
    
    def __init__(self):
        self.mystem = Mystem()


    def clean(self, text):
        '''Does string cleaning and text normalization
        text    -> string to clean
        returns -> lemmatized and tokenized list of strings without stopwords and special symbols
        '''

        # Lemmatizes tokens
        tokens = self.mystem.lemmatize(text)
        
        # Returns a list of tokens
        return simple_preprocess(' '.join(tokens))