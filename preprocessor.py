import re
import string
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
# Other choices possible, evaluation to be done,

class Preprocessor:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print('Downloading required nltk resources. This will only happen once.')
            nltk.download('punkt')
            nltk.download('wordnet')
        self._tokenizer = word_tokenize
        self._lemmatizer = WordNetLemmatizer()
        
    def process(self,inputString):
        # some hephen-connected words need to be taken care of
        cleanString = self.stringLower(inputString).replace("-", " ")
        cleanString = self.stringNumRemover(cleanString)
        cleanString = self.stringPunRemover(cleanString)
        wordList = self.stringSwRemover(cleanString)
        return [self.wordLemmatizer(word) for word in wordList]
    
    def stringTokenize(self,inputString):
        return self._tokenizer(inputString)
    
    def stringLower(self,inputString):
        return inputString.lower()
    
    def wordLemmatizer(self,inputWord):
        return self._lemmatizer.lemmatize(inputWord)
    
    def stringSwRemover(self,inputString):
        '''
        remove stop words, return a list
        '''
        return [token for token in self._tokenizer(inputString) if token not in ENGLISH_STOP_WORDS]
        
    
    def stringNumRemover(self,inputString):
        '''
        remove numbers
        '''
        return re.sub(r'\d+', '', inputString)
    
    def stringPunRemover(self,inputString):
        '''
        remove punctuation
        '''
        inputString = inputString.strip()
        return inputString.translate(inputString.maketrans("","", '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))
    