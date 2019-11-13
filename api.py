'''
api interface to provide the required function inquiries
'''
import random
import json
import numpy as np
from collections import Counter
import os
import pickle
from model import Model, word2vecEstimator, doc2vecEstimator
from dataPreparer import generateUIDs, splitDataSets, initPrepare
from preprocessor import Preprocessor
from gensim.models import Word2Vec
from Levenshtein import distance as lev_distance
import gensim.downloader as gensim_api

def dataSaver():
    """This function prepares and save doc2vec, word2vec and glove vectors to directory ./data. It also saves the trained model for fast loading when use. Note theoretically the model needs to be trained by trainCV() methods. This is very computationally expensive. The train is therefore explicitly done using the whole datasets without CV.
    """
    _, uids, COURSENAMES, COURSEDESCRIPTIONS, TRUTHTABLE = initPrepare()
    random.shuffle(uids)
    try:
        os.mkdir("data", 755)
    except:
        print("data directory already existis.")
    with open('data/coursedescriptions.json', 'w') as outfile:
        json.dump(COURSEDESCRIPTIONS, outfile)
    with open('data/pseudoTruthTable.json', 'w') as outfile:
        json.dump(TRUTHTABLE, outfile)
    with open('data/coursenames.json', 'w') as outfile:
        json.dump(COURSENAMES, outfile)
    # generate best word2vec model, best doc2vec model and saves them
    w2vModel = Model("word2vec")
    d2vModel = Model("doc2vec")
    random.seed(2019)
    # glove model mean vector saves
    gloveModel = gensim_api.load("word2vec-google-news-300")
    p1 = Preprocessor()
    # use the whole data sets can take extremely long time
    # w2vModel.trainCV(uids)
    # d2vModel.trainCV(uids)
    # the following training methods is solely for performance reasons.
    uids, desc1, desc2, _, _ = w2vModel.augmentData(uids)  # any model works
    # cross validation on the word2vecEstimator

    sentences1 = [p1.process(desc) for desc in desc1]
    sentences2 = [p1.process(desc) for desc in desc2]
    w2vModel.vecModelEstimator = word2vecEstimator(TRUTHTABLE)

    w2vModel.vecModelEstimator.fit(uids, [(sentences1[i], sentences2[i])
                                          for i in range(len(uids))])

    d2vModel.vecModelEstimator = doc2vecEstimator(TRUTHTABLE)
    d2vModel.vecModelEstimator.fit(uids, [(sentences1[i], sentences2[i])
                                          for i in range(len(uids))])

    word2vecJson = {}
    glove2vecJson = {}
    doc2vecJson = {}
    for cid, words in COURSEDESCRIPTIONS.items():
        vec3 = d2vModel.vecModelEstimator.model.infer_vector(
            p1.process(words))
        doc2vecJson[cid] = list(map(lambda x: str(x), vec3))
        vecs1, vecs2 = [], []
        for word in p1.process(words):
            try:
                vecs1.append(w2vModel.vecModelEstimator.model[word])
            except:
                print(word + " is not in trained word2vec model, skipped.")
            try:
                vecs2.append(gloveModel[word])
            except:
                print(word + " is not in glove model, skipped.")
        if len(vecs2) != 0:
            vec2 = np.mean(np.array(vecs2), axis=0)
            glove2vecJson[cid] = list(map(lambda x: str(x), vec2))
        if len(vecs1) != 0:
            vec1 = np.mean(np.array(vecs1), axis=0)
            word2vecJson[cid] = list(map(lambda x: str(x), vec1))

    with open('data/bestword2vec.json', 'w') as outfile:
        json.dump(word2vecJson, outfile)
    with open('data/glove2vec.json', 'w') as outfile:
        json.dump(glove2vecJson, outfile)
    with open('data/bestdoc2vec.json', 'w') as outfile:
        json.dump(doc2vecJson, outfile)

    # pickle best models and save them
    with open("data/bestword2vec.pickle", "wb") as outputfile:
        pickle.dump(w2vModel, outputfile)
    with open("data/bestdoc2vec.pickle", "wb") as outputfile:
        pickle.dump(d2vModel, outputfile)


def uclaDataSaver():
    """ Given the model trained by api.dataSaver() and ucla courses parsed by crawler.uclaParse(), api.uclaDataSaver() prepares and saves doc2vec, word2vec and glove vector of ucla courses locally for fast loading.
    """
    try:
        with open("data/bestword2vec.pickle", "rb") as fp:
            w2vModel = pickle.load(fp)
        with open("data/bestdoc2vec.pickle", "rb") as fp:
            d2vModel = pickle.load(fp)
    except:
        print("Please run dataSaver() first to obtain model trained on USC courses")
        return
    gloveModel = gensim_api.load("word2vec-google-news-300")
    p1 = Preprocessor()
    try:
        with open("data/uclacourses.json", "r") as fp:
            UCLACOURSES = json.load(fp)
    except:
        print("Please run crawler.uclaDataSaver() to obtain uclaCourses.json")
        return

    uclaword2vecJson = {}
    uclaglove2vecJson = {}
    ucladoc2vecJson = {}
    for cid, course in UCLACOURSES.items():
        words = course["desc"]
        vec3 = d2vModel.vecModelEstimator.model.infer_vector(
            p1.process(words))
        ucladoc2vecJson[cid] = list(map(lambda x: str(x), vec3))
        vecs1, vecs2 = [], []
        for word in p1.process(words):
            try:
                vecs1.append(w2vModel.vecModelEstimator.model[word])
            except:
                print(word + " is not in trained word2vec model, skipped.")
            try:
                vecs2.append(gloveModel[word])
            except:
                print(word + " is not in glove model, skipped.")
        if len(vecs2) != 0:
            vec2 = np.mean(np.array(vecs2), axis=0)
            uclaglove2vecJson[cid] = list(map(lambda x: str(x), vec2))
        if len(vecs1) != 0:
            vec1 = np.mean(np.array(vecs1), axis=0)
            uclaword2vecJson[cid] = list(map(lambda x: str(x), vec1))

    with open('data/uclabestword2vec.json', 'w') as outfile:
        json.dump(uclaword2vecJson, outfile)
    with open('data/uclaglove2vec.json', 'w') as outfile:
        json.dump(uclaglove2vecJson, outfile)
    with open('data/uclabestdoc2vec.json', 'w') as outfile:
        json.dump(ucladoc2vecJson, outfile)


class API:
    """API is the exposed objects for end user to interact with the application.

    :param model_algo: one of the five algorithms. The default is word2vec
    :type model_algo: string.
    """

    def __init__(self, model_algo="word2vec"):
        """Constructor method. Note 'glove' model may require a very long time to load on laptop, therefore is not recommended.

        """
        if model_algo == "word2vec":
            with open("data/bestword2vec.pickle", "rb") as fp:
                self.model = pickle.load(fp)
        elif model_algo == "doc2vec":
            with open("data/bestdoc2vec.pickle", "rb") as fp:
                self.model = pickle.load(fp)
        elif model_algo == "glove":
            self.model = Model(algo="glove")
        else:
            self.model = Model(algo=model_algo)
        self.preprocessor = Preprocessor()

    def stringQuery(self, inputString, verbose=False):
        """Query with a string to obtain top 10 most related **USC** courses with respect to the input string.

        :param inputString: the string to query with
        :type inputString: string
        :param verbose: If true, output unseen word in the prediction steps
        :type verbose: bool

        :return: A 10-element list of tuples. each tuple contains the course id and corresponding scores. Note the scores' absolute values can't be compared across chosen algorithms in principle.
        :rtype: list

        """
        if self.model.algo in self.model.trainable:
            if self.model.algo == "word2vec":
                words = self.preprocessor.process(inputString)
                vecs = []
                for word in words:
                    try:
                        vecs.append(self.model.vecModelEstimator.model[word])
                    except KeyError:
                        if verbose:
                            print(
                                word + " is not learned in train data set, ignored.")
                        continue
                if len(vecs) == 0:
                    if verbose:
                        print("The input string contains no seen words. Return None.")
                    return None
                vec = np.mean(np.array(vecs), axis=0)
                vec = (vec - self.model.vecModelEstimator.ave) / \
                    self.model.vecModelEstimator.std
                # can keep the object persistent in objects TODO
                with open("data/bestword2vec.json", "r") as fp:
                    courseWordVecs = json.load(fp)
                for key in courseWordVecs.keys():
                    courseWordVecs[key] = list(
                        map(lambda x: float(x), courseWordVecs[key]))
                scores = [(cid, np.dot(vec, tmpVec)/np.linalg.norm(vec)/np.linalg.norm(tmpVec))
                          for cid, tmpVec in courseWordVecs.items()]
                return sorted(scores, key=lambda pair: -pair[1])[:10]

            elif self.model.algo == "doc2vec":
                words = self.preprocessor.process(inputString)
                vec = self.model.vecModelEstimator.model.infer_vector(words)
                vec = (vec - self.model.vecModelEstimator.ave) / \
                    self.model.vecModelEstimator.std
                # can keep the object persistent in objects TODO
                with open("data/bestdoc2vec.json", "r") as fp:
                    courseDocVecs = json.load(fp)
                for key in courseDocVecs.keys():
                    courseDocVecs[key] = list(
                        map(lambda x: float(x), courseDocVecs[key]))
                scores = [(cid, np.dot(vec, tmpVec)/np.linalg.norm(vec)/np.linalg.norm(tmpVec))
                          for cid, tmpVec in courseDocVecs.items()]
                return sorted(scores, key=lambda pair: -pair[1])[:10]
        else:
            # non-trainable model, has to loop over one by one
            if self.model.algo == "Levenshtein":
                print(
                    "Warning: the Levenshtein algorithm now treats the input string as name of courses.")
                with open("data/coursenames.json", "r") as fp:
                    COURSENAMES = json.load(fp)
                scores = []
                for cid, name2 in COURSENAMES.items():
                    scores.append(
                        (cid, lev_distance(inputString, name2)/max(len(inputString), len(name2))))
                return sorted(scores, key=lambda pair: -pair[1])[:10]
            else:
                with open("data/coursedescriptions.json", "r") as fp:
                    COURSEDESCRIPTIONS = json.load(fp)
                words1 = self.preprocessor.process(inputString)
                if self.model.algo == "Jaccard":
                    scores = []
                    for cid, desc in COURSEDESCRIPTIONS.items():
                        words2 = self.preprocessor.process(desc)
                        scores.append(
                            (cid, self.model.Jaccard(words1, words2)))

                    return sorted(scores, key=lambda pair: -pair[1])[:10]

                if self.model.algo == "glove":
                    with open('data/glove2vec.json', 'r') as fp:
                        courseGloveVecs = json.load(fp)
                    for key in courseGloveVecs.keys():
                        courseGloveVecs[key] = list(
                            map(lambda x: float(x), courseGloveVecs[key]))
                    vecs = []
                    for word in words1:
                        try:
                            tmpvec = self.model.gloveModel[word]
                            vecs.append(tmpvec)
                        except KeyError:
                            if verbose:
                                print(word + " is not in glove model, skipped.")
                            continue
                    if len(vecs) != 0:
                        vec = np.mean(np.array(vecs), axis=0)
                        scores = [(cid, np.dot(vec, tmpVec)/np.linalg.norm(vec)/np.linalg.norm(tmpVec))
                                  for cid, tmpVec in courseGloveVecs.items()]
                        return sorted(scores, key=lambda pair: -pair[1])[:10]
                else:
                    print(
                        inputString + " doesn't contain word in glove model vocabulary. This is strange. Please check.")
                    return None

    def queryPrereqsCIDs(self, cids):
        """find the number of the prerequisites courses(cids) input is a list of cids that are in the course(cid)'s top-3 related courses.

        :param cids: a list of **USC** course ids
        :type cids: list of string
        :return: a list of integer with the same lengh of input, each representing the number of prerequisites courses that are in the top-3 related courses of the corresponding course. Note, if a course id doesen't exist in the database, the value will be -1, it is the user's responsibility to check it.
        :rtype: list
        """
        '''
        
        '''
        with open("data/courses.json", "r") as fp:
            COURSES = json.load(fp)
        res = []
        for cid in cids:
            if cid not in COURSES:
                res.append(-1)
                continue
            if len(COURSES[cid]["prereqs"]) == 0:
                # cid has no prereqsites courses
                print(cid + " has no prerequsites.")
                res.append(0)
            else:
                print(cid + " has prerequisites: ", COURSES[cid]["prereqs"])
                top3ID = self._queryCID(COURSES, cid, 3)
                overlap = 0
                for prereq in COURSES[cid]["prereqs"]:
                    if prereq in top3ID:
                        overlap += 1
                res.append(overlap)
        return res

    def querySameSchoolCIDs(self, cids):
        """Find the ratio of same school courses in the top 10 related courses for each course id in the parameter cids.

        :param cids: a list of **USC** course ids
        :type cids: list of string
        :return: the corresponding same-school course ratio for each course id in cids. Note, if a course id doesen't exist in the database, the value will be -1, it is the user's responsibility to check it.
        :rtype: a list of float in [0,1]
        """
        with open("data/courses.json", "r") as fp:
            COURSES = json.load(fp)
        res = []
        for cid in cids:
            if cid not in COURSES:
                res.append(-1)
                continue
            school = COURSES[cid]["school"]
            top10ID = self._queryCID(COURSES, cid, 10)
            print("Top10 related courses of "+cid + ": ", top10ID)
            ratio = np.mean([COURSES[rel_cid]["school"]
                             == school for rel_cid in top10ID])
            res.append(ratio)
        return res

    def _queryCID(self, COURSES, cid, k):
        """Query top [k] **USC** courses which are relevant to course [cid]

        :param COURSES: the dictionary that stores the USC course information
        :type COURSES: dictionary
        :param cid: course id
        :type cid: string
        :param k: number of returned relevant courses
        :type k: integer
        :return: course ids of relevant courses
        :rtype: k-element list of string
        """

        top = self._queryTopKCID(cid=cid, k=k, COURSES=COURSES)
        topID = [c[0] for c in top]
        return topID

    def _queryTopKCID(self, cid, k, COURSES):
        """helper function of _queryCID()
        """
        assert(k <= 10)
        if self.model.algo == "Levenshtein":
            string = COURSES[cid]["name"]
        else:
            string = COURSES[cid]["desc"]
        return self.stringQuery(string)[:k]

    def queryUCLACID(self, cid):
        """Query UCLA computer science course given a course id of USC courses. Based on the link provided, only courses from the computer science department are considered. This method has logic overlap with queryString() but queryString() only allows quering on USC courses. In the future, the two methods may merge.
        :param cid: course ID of a USC course
        :type cid: string
        :return: tuple of course id and course name of counterpart UCLA course
        :rtype: 2-element tuple of strings
        """
        try:
            # reading ucla course info and vector representations of courses
            with open("data/uclacourses.json", "r") as fp:
                uclaCourses = json.load(fp)
        except:
            print(
                "Please run API().uclaDataSaver() first to obtain vector representation of the data.")
            return
        # can be persistent

        if self.model.algo == "Levenshtein":
            with open("data/coursenames.json", "r") as fp:
                COURSENAMES = json.load(fp)
            uscName = COURSENAMES[cid]
            resdist, resId, resName = 10000, None, None  # magic number
            for uclaID in uclaCourses.keys():
                uclaName = uclaCourses[uclaID]["name"]
                dist = lev_distance(uscName, uclaName) / \
                    max(len(uscName), len(uclaName))
                if dist < resdist:
                    resdist = dist
                    resId = uclaID
                    resName = uclaName
            return (resId, resName)
        # the rest needs descriptions
        with open("data/coursedescriptions.json", "r") as fp:
            COURSEDESCRIPTIONS = json.load(fp)
        p1 = Preprocessor()
        if self.model.algo == "Jaccard":
            uscDesc = p1.process(COURSEDESCRIPTIONS[cid])
            resScore, resId, resName = -1, None, None
            for uclaID in uclaCourses.keys():
                uclaDesc = p1.process(uclaCourses[uclaID]["desc"])
                score = _Jaccard(uscDesc, uclaDesc)
                if score > resScore:
                    resScore > score
                    resId = uclaID
                    resName = uclaCourses[uclaID]["name"]
            return (resId, resName)

        elif self.model.algo == "word2vec":
            with open("data/bestword2vec.json", "r") as fp:
                uscWord2Vec = json.load(fp)
            with open("data/uclabestword2vec.json", "r") as fp:
                uclaWord2Vec = json.load(fp)
            uscVec = [float(x) for x in uscWord2Vec[cid]]
            resScore, resId, resName = -1, None, None  # magic number
            for uclaID, uclaVec in uclaWord2Vec.items():
                uclaVec = [float(x) for x in uclaVec]
                score = np.dot(uscVec, uclaVec) / \
                    np.linalg.norm(uclaVec)/np.linalg.norm(uscVec)
                if score > resScore:
                    resScore = score
                    resId = uclaID
                    resName = uclaCourses[uclaID]["name"]
            return (resId, resName)

        elif self.model.algo == "doc2vec":
            with open("data/bestdoc2vec.json", "r") as fp:
                uscDoc2Vec = json.load(fp)
            with open("data/uclabestdoc2vec.json", "r") as fp:
                uclaDoc2Vec = json.load(fp)
            uscVec = [float(x) for x in uscDoc2Vec[cid]]
            resScore, resId, resName = -1, None, None  # magic number
            for uclaID, uclaVec in uclaDoc2Vec.items():
                uclaVec = [float(x) for x in uclaVec]
                score = np.dot(uscVec, uclaVec) / \
                    np.linalg.norm(uclaVec)/np.linalg.norm(uscVec)
                if score > resScore:
                    resScore = score
                    resId = uclaID
                    resName = uclaCourses[uclaID]["name"]
            return (resId, resName)

        else:
            assert(self.model.algo == "glove")
            with open("data/uclaglove2vec.json", "r") as fp:
                uclaGlove = json.load(fp)
            with open("data/glove2vec.json", "r") as fp:
                uscGlove = json.load(fp)
            uscVec = [float(x) for x in uscGlove[cid]]
            resScore, resId, resName = -1, None, None  # magic number
            for uclaID, uclaVec in uclaGlove.items():
                uclaVec = [float(x) for x in uclaVec]
                score = np.dot(uscVec, uclaVec) / \
                    np.linalg.norm(uclaVec)/np.linalg.norm(uscVec)
                if score > resScore:
                    resScore = score
                    resId = uclaID
                    resName = uclaCourses[uclaID]["name"]
            return (resId, resName)


def _Jaccard(ls1, ls2):
    """Return the Jaccard similarity of two lists

    :param ls1: first list, element can be of any kind.
    :type ls1: list
    :param ls2: second list, element can be of any kind.
    :type ls2: list
    :return: a ratio indicating the similarity between ls1 and ls2, 0 means no similarity at all, 1 means identical (in the orderless sense).
    :rtype: float
    """
    interNum = len(list((Counter(ls1) & Counter(ls2)).elements()))
    return interNum/(len(ls1)+len(ls2)-interNum)
