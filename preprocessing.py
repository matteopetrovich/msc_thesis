import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer
import string

class PREPROCESSING():

    def process(text):
    
        text = re.sub('[^A-Za-z0-9]+', ' ',  text)
        tokens = word_tokenize(text)
        tokens = [w.lower() for w in tokens]
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        return words
    
    
    def process_stop(text):
    
        text = re.sub('[^A-Za-z0-9]+', ' ',  text)
        tokens = word_tokenize(text)
        tokens = [w.lower() for w in tokens]
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        return words
    
    
    def process_lancaster(text):
    
        text = re.sub('[^A-Za-z0-9]+', ' ',  text)
        tokens = word_tokenize(text)
        tokens = [w.lower() for w in tokens]
        stemmer = LancasterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        return words
    
    
    def process_lancaster_stop(text):
    
        text = re.sub('[^A-Za-z0-9]+', ' ',  text)
        tokens = word_tokenize(text)
        tokens = [w.lower() for w in tokens]
        stemmer = LancasterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        return words
    
    def process_lemmatizer(text):
    
        text = re.sub('[^A-Za-z0-9]+', ' ',  text)
        tokens = word_tokenize(text)
        tokens = [w.lower() for w in tokens]
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        return words
    
    
    def process_lemmatizer_stop(text):
    
        text = re.sub('[^A-Za-z0-9]+', ' ',  text)
        tokens = word_tokenize(text)
        tokens = [w.lower() for w in tokens]
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        return words