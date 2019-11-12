import re
import string
from gensim.models import KeyedVectors
import gensim.downloader as api
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from Levenshtein import distance as lev_distance
from preprocessor import Preprocessor
from dataPreparer import initPrepare



class word2vecEstimator(BaseEstimator):
    def __init__(self, TRUTHTABLE, size=100, alpha=0.025, window=5, sg=0):
        '''
        Initialize parameters to perform grid search
        It is possible to further extend the list of parameters.
        '''
        self.TRUTHTABLE = TRUTHTABLE
        self.model = None
        self.size = size  # vector size
        self.alpha = alpha  # learning rate
        self.window = window  # maximal distance
        self.sg = sg
        # train data information
        self.ave, self.std = None, None
        # to augment the values to the same domain
        self.minVal = np.min(list(self.TRUTHTABLE.values()))
        self.maxVal = np.max(list(self.TRUTHTABLE.values()))

    def fit(self, trainUIDs, sentlists, verbose=True):
        '''
        targetless training
        '''
        sentences1 = [sentlists[i][0] for i in range(len(sentlists))]
        sentences2 = [sentlists[i][1] for i in range(len(sentlists))]
        self.model = Word2Vec(sentences1 + sentences2,
                              seed=2019,
                              size=self.size,
                              alpha=self.alpha,
                              window=self.window,
                              sg=self.sg,
                              min_count=1
                              )
        # obtain normalized information to avoid data leaking
        # in fit, ground truth can be peaked
        scores = []
        for sent1, sent2 in tqdm(zip(sentences1, sentences2)):
            vecs1, vecs2 = [], []
            for word in sent1:
                try:
                    tmp = self.model[word]
                    vecs1.append(tmp)
                except KeyError:
                    if verbose:
                        print(word + " is not learned.")
                    continue
            for word in sent2:
                try:
                    tmp = self.model[word]
                    vecs2.append(tmp)
                except KeyError:
                    if verbose:
                        print(word + " is not learned.")
                    continue
            vecs1 = np.array(vecs1)
            vecs2 = np.array(vecs2)
            vec1 = np.mean(vecs1, axis=0)
            vec2 = np.mean(vecs2, axis=0)
            scores.append(np.dot(vec1, vec2) /
                          np.linalg.norm(vec1)/np.linalg.norm(vec2))
        self.ave = np.mean(scores)
        self.std = max(np.abs(scores))/max(abs(self.minVal), abs(self.maxVal))

    def predict(self, testUIDs, sentlists, verbose=False):
        '''
        1. obtain vector
        2. compare cosine similarity
        '''
        try:
            assert(self.model != None)
        except:
            print("Model needs to be trained first.")
            return
        scores = []
        sentences1 = [sentlists[i][0] for i in range(len(sentlists))]
        sentences2 = [sentlists[i][1] for i in range(len(sentlists))]
        for sent1, sent2 in zip(sentences1, sentences2):
            vecs1, vecs2 = [], []
            for word in sent1:
                try:
                    vecs1.append(self.model[word])
                except KeyError:
                    if verbose:
                        print(word + " is not learned in train data set, ignored.")
                    continue
            for word in sent2:
                try:
                    vecs2.append(self.model[word])
                except KeyError:
                    if verbose:
                        print(word + " is not learned in train data set, ignored.")
                    continue
            vecs1 = np.array(vecs1)
            vecs2 = np.array(vecs2)
            if len(vecs1) == 0 or len(vecs2) == 0:
                if verbose:
                    print("One of ", sent1, sent2,
                          "contains no seen words. Assigning score 0.")
                scores.append(0)
                continue
            vec1 = np.mean(vecs1, axis=0)
            vec2 = np.mean(vecs2, axis=0)
            # print(vec1.shape,vec2.shape,np.dot(vec1, vec2),np.linalg.norm(vec1),np.linalg.norm(vec2))
            score = np.dot(vec1, vec2) / np.linalg.norm(vec1) / \
                np.linalg.norm(vec2)
            scores.append(score)
        # output normalized score
        return [(score-self.ave)/self.std for score in scores]

    def score(self, uids, sentlists):
        '''
        return a score to compare with the TRUTHTABLE
        '''
        truths = [self.TRUTHTABLE[uid] for uid in uids]
        scores = self.predict(uids, sentlists)
        # print("scores",scores)
        min_error = np.sum(
            [(truths[i]-scores[i])**2 for i in range(len(uids))])/len(uids)
        return -1 * min_error  # the higher the better


class doc2vecEstimator(BaseEstimator):
    def __init__(self, TRUTHTABLE, dm=0, vector_size=50, window=5, alpha=0.025):
        self.TRUTHTABLE = TRUTHTABLE
        self.model = None
        self.dm = dm
        self.vector_size = vector_size
        self.window = window
        self.alpha = alpha
        self.ave, self.std = None, None
        # to augment the values to the same domain
        self.minVal = np.min(list(self.TRUTHTABLE.values()))
        self.maxVal = np.max(list(self.TRUTHTABLE.values()))

    def fit(self, trainUIDs, sentlists, verbose=True):
        sentences1 = [sentlists[i][0] for i in range(len(sentlists))]
        sentences2 = [sentlists[i][1] for i in range(len(sentlists))]
        documents = [TaggedDocument(doc, [i])
                     for i, doc in enumerate(sentences1 + sentences2)]
        self.model = Doc2Vec(documents,
                             seed=2019,
                             dm=self.dm,
                             vector_size=self.vector_size,
                             window=self.window,
                             alpha=self.alpha
                             )
        scores = []
        for sent1, sent2 in zip(sentences1, sentences2):
            vec1 = self.model.infer_vector(sent1)
            vec2 = self.model.infer_vector(sent2)
            score = np.dot(vec1, vec2)/np.linalg.norm(vec1) / \
                np.linalg.norm(vec2)
            scores.append(score)
        self.ave = np.mean(scores)
        self.std = max(np.abs(scores))/max(abs(self.minVal), abs(self.maxVal))

    def predict(self, testUIDs, sentlists, verbose=False):
        try:
            assert(self.model != None)
        except:
            print("Model needs to be trained first.")
            return
        sentences1 = [sentlists[i][0] for i in range(len(sentlists))]
        sentences2 = [sentlists[i][1] for i in range(len(sentlists))]
        scores = []
        for sent1, sent2 in zip(sentences1, sentences2):
            vec1 = self.model.infer_vector(sent1)
            vec2 = self.model.infer_vector(sent2)
            score = np.dot(vec1, vec2)/np.linalg.norm(vec1) / \
                np.linalg.norm(vec2)
            scores.append(score)
        return [(score-self.ave)/self.std for score in scores]

    def score(self, uids, sentlists):
        truths = [self.TRUTHTABLE[uid] for uid in uids]
        scores = self.predict(uids, sentlists)
        min_error = np.sum(
            [(truths[i]-scores[i])**2 for i in range(len(uids))])/len(uids)
        return -1 * min_error  # the higher the better


class Model:
    def __init__(self, algo="Jaccard"):
        self.algo = algo
        if self.algo == "glove":
            print("Loading model can take long time. Please be paitent.")
            self.gloveModel = api.load("word2vec-google-news-300")
        self.simAlgo = {"Jaccard": None,
                        "Levenshtein": None,
                        "word2vec": None,
                        "doc2vec": None,
                        "glove": None
                        }
        self.trainable = {
            "word2vec": True,
            "doc2vec": True,
            # https://radimrehurek.com/gensim/models/keyedvectors.html
            # Vectors exported by the Facebook and Google tools do not support further training,
            # but you can still load them into KeyedVectors
        }
        self.bestParam = {}
        self.preprocessor = Preprocessor()
        self.vecModelEstimator = None

        try:
            with open("data/coursedescriptions.json", "r") as inputfile:
                self.COURSEDESCRIPTIONS = json.load(inputfile)
            with open("data/coursenames.json", "r") as inputfile:
                self.COURSENAMES = json.load(inputfile)
            with open("data/pseudoTruthTable.json", "r") as inputfile:
                self.TRUTHTABLE = json.load(inputfile)
                for key in self.TRUTHTABLE.keys():
                    self.TRUTHTABLE[key] = float(self.TRUTHTABLE[key])
        except:
            _, _, self.COURSENAMES, self.COURSEDESCRIPTIONS, self.TRUTHTABLE = initPrepare()

    def trainCV(self, trainUIDs, verbose=False):
        '''
        '''
        if self.algo not in self.trainable:
            print(self.algo, " algorithm is not trainable. Try 'word2vec' or 'doc2vec'.")
            return
        trainUIDs, desc1, desc2, _, _ = self.augmentData(trainUIDs)
        # cross validation on the word2vecEstimator
        sentences1 = [self.preprocessor.process(desc) for desc in desc1]
        sentences2 = [self.preprocessor.process(desc) for desc in desc2]
        if self.algo == "word2vec":
            tuned_params = {"size": [50, 20, 10],
                            "alpha": [0.010, 0.025],
                            "window": [2, 5],
                            "sg": [0, 1]}
            gs = GridSearchCV(word2vecEstimator(
                self.TRUTHTABLE), tuned_params, cv=3)
            gs.fit(trainUIDs, [(sentences1[i], sentences2[i])
                               for i in range(len(trainUIDs))])
            self.bestParam = gs.best_params_
            self.vecModelEstimator = gs.best_estimator_
            # print("CV results:", gs.cv_results_)
        if self.algo == "doc2vec":
            tuned_params = {"dm": [0, 1],
                            "vector_size": [50, 20, 10],
                            "alpha": [0.010, 0.025],
                            "window": [2, 5]}
            gs = GridSearchCV(doc2vecEstimator(
                self.TRUTHTABLE), tuned_params, cv=3)
            gs.fit(trainUIDs, [(sentences1[i], sentences2[i])
                               for i in range(len(trainUIDs))])
            self.bestParam = gs.best_params_
            self.vecModelEstimator = gs.best_estimator_
            print("CV results:", gs.cv_results_)

    def predict(self, testUIDs, verbose=False):
        '''
        return a 1*n list of float
        '''

        testUIDs, desc1, desc2, name1, name2 = self.augmentData(testUIDs)
        scores = []
        if self.algo not in self.trainable:
            print("Warning: The scores returned by untrainable algorithm " +
                  self.algo + " are not numerically comparable with the pseudo groundtruth.")

        if self.algo == "Jaccard":
            for idx in range(len(testUIDs)):
                words1 = self.preprocessor.process(desc1[idx])
                words2 = self.preprocessor.process(desc2[idx])
                scores.append(self.Jaccard(words1, words2))
            return self.zero(scores)

        if self.algo == "Levenshtein":
            for idx in range(len(testUIDs)):
                scores.append(self.Levenshtein(name1[idx], name2[idx]))
            return self.zero(scores)

        if self.algo == "glove":
            for idx in range(len(testUIDs)):
                words1 = self.preprocessor.process(desc1[idx])
                words2 = self.preprocessor.process(desc2[idx])
                vecs1, vecs2 = [], []
                for word in words1:
                    try:
                        vecs1.append(self.gloveModel.word_vec(word))
                    except KeyError:
                        if verbose:
                            print("'"+word+"'" +
                                  " is not in glove model. It is ignored.")
                        continue
                for word in words2:
                    try:
                        vecs2.append(self.gloveModel.word_vec(word))
                    except KeyError:
                        if verbose:
                            print("'"+word+"'" +
                                  " is not in glove model. It is ignored.")
                        continue
                vecs1 = np.array(vecs1)
                vecs2 = np.array(vecs2)
                if len(vecs1) == 0 or len(vecs2) == 0:
                    if verbose:
                        print(
                            "One of the sentences contains no seen words. Assigning score 0.")
                    scores.append(0)
                    continue
                vec1 = np.mean(vecs1, axis=0)
                vec2 = np.mean(vecs2, axis=0)
                scores.append(
                    self.gloveModel.cosine_similarities(vec1, [vec2])[0])
            return self.zero(scores)

        if self.algo == "word2vec":
            try:
                assert(self.vecModelEstimator != None)
            except:
                print("Train the model first with trainCV().")
                return
            sentences1 = [self.preprocessor.process(desc) for desc in desc1]
            sentences2 = [self.preprocessor.process(desc) for desc in desc2]
            return self.vecModelEstimator.predict(testUIDs, [(sentences1[i], sentences2[i])
                                                             for i in range(len(testUIDs))])
        if self.algo == "doc2vec":
            try:
                assert(self.vecModelEstimator != None)
            except:
                print("Train the model first with trainCV().")
                return
            sentences1 = [self.preprocessor.process(desc) for desc in desc1]
            sentences2 = [self.preprocessor.process(desc) for desc in desc2]
            return self.vecModelEstimator.predict(testUIDs, [(sentences1[i], sentences2[i])
                                                             for i in range(len(testUIDs))])

    def Jaccard(self, ls1, ls2):
        interNum = len(list((Counter(ls1) & Counter(ls2)).elements()))
        return interNum/(len(ls1)+len(ls2)-interNum)

    def Levenshtein(self, name1, name2):
        '''
        https://rawgit.com/ztane/python-Levenshtein/master/docs/Levenshtein.html#Levenshtein-distance
        '''
        return lev_distance(name1, name2)/max(len(name1), len(name2))

    def augmentData(self, uids):
        desc1, desc2 = [], []
        name1, name2 = [], []
        for uid in uids:
            twoids = uid.split("-")
            cid1, cid2 = twoids[0], twoids[1]
            desc1.append(self.COURSEDESCRIPTIONS[cid1])
            desc2.append(self.COURSEDESCRIPTIONS[cid2])
            name1.append(self.COURSENAMES[cid1])
            name2.append(self.COURSENAMES[cid2])
        return uids, desc1, desc2, name1, name2

    def zero(self, scores):
        ave = np.mean(scores)
        std = np.std(scores)
        self.bestParam["ave"] = ave
        self.bestParam["std"] = std
        # linear map to
        return [(x - ave)/std for x in scores]
