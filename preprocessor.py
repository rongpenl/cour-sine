import re
import string
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

class Preprocessor:
    """Preprocessor class. It can performs various kinds of class but the most important interface is the process() method. 
    """
    def __init__(self):
        """Constructor method
        """
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
        """process() is the main interface. The method 1. lower the string 2. replace hephen by space 3. remove number 4. remove punctuations 5. remove stop words 6. lemmatize the tokenized words.
        
        :param inputString: string to process
        :type inputString: string
        :return: list of cleaned words
        :rtype: list 
        """
        # some hephen-connected words need to be taken care of
        cleanString = self.stringLower(inputString).replace("-", " ")
        cleanString = self.stringNumRemover(cleanString)
        cleanString = self.stringPunRemover(cleanString)
        wordList = self.stringSwRemover(cleanString)
        return [self.wordLemmatizer(word) for word in wordList]
    
    def stringTokenize(self,inputString):
        """Tokenize the input string.
        
        :param inputString: string to be tokenized.
        :type inputString: string
        :return: list of words
        :rtype: list
        """
        return self._tokenizer(inputString)
    
    def stringLower(self,inputString):
        """Lower the string.
        """
        return inputString.lower()
    
    def wordLemmatizer(self,inputWord):
        """lemmatize the input word. For instance, better -> good.
        
        :param inputWord: a word
        :type inputWord: string
        :return: lemmatized word
        :rtype: string
        """
        return self._lemmatizer.lemmatize(inputWord)
    
    def stringSwRemover(self,inputString):
        '''remove stop words, return a list
        '''
        return [token for token in self._tokenizer(inputString) if token not in ENGLISH_STOP_WORDS]
        
    
    def stringNumRemover(self,inputString):
        '''remove numbers
        '''
        return re.sub(r'\d+', '', inputString)
    
    def stringPunRemover(self,inputString):
        '''remove punctuation words
        '''
        inputString = inputString.strip()
        return inputString.translate(inputString.maketrans("","", '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))
    