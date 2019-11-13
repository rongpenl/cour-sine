'''
api interface to provide the required function inquiries
'''
import random
import json
import numpy as np
from model import Model, word2vecEstimator, doc2vecEstimator
from dataPreparer import generateUIDs, splitDataSets, initPrepare
from preprocessor import Preprocessor
from gensim.models import Word2Vec
from Levenshtein import distance as lev_distance
import gensim.downloader as gensim_api
import os
import pickle


def dataSaver():
    '''
    A time consuming function that generates all useful datasets and save them locally
    '''
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
    tmpModel = Model("word2vec")
    tmpModel2 = Model("doc2vec")
    random.seed(2019)
    # glove model mean vector saves
    gloveModel = gensim_api.load("word2vec-google-news-300")
    p1 = Preprocessor()
    # use the whole data sets can take extremely long time
    # tmpModel.trainCV(uids)
    # tmpModel2.trainCV(uids)
    # the following training methods is solely for performance reasons.
    uids, desc1, desc2, _, _ = tmpModel.augmentData(uids)  # any model works
    # cross validation on the word2vecEstimator
    sentences1 = [p1.process(desc) for desc in desc1]
    sentences2 = [p1.process(desc) for desc in desc2]
    tmpModel.vecModelEstimator = word2vecEstimator(TRUTHTABLE)

    tmpModel.vecModelEstimator.fit(uids, [(sentences1[i], sentences2[i])
                                          for i in range(len(uids))])

    tmpModel2.vecModelEstimator = doc2vecEstimator(TRUTHTABLE)
    tmpModel2.vecModelEstimator.fit(uids, [(sentences1[i], sentences2[i])
                                           for i in range(len(uids))])

    word2vecJson = {}
    glove2vecJson = {}
    doc2vecJson = {}
    for cid, words in COURSEDESCRIPTIONS.items():
        vec3 = tmpModel2.vecModelEstimator.model.infer_vector(
            p1.process(words))
        doc2vecJson[cid] = list(map(lambda x: str(x), vec3))
        vecs1, vecs2 = [], []
        for word in p1.process(words):
            try:
                vecs1.append(tmpModel.vecModelEstimator.model[word])
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

    # pickle best models
    with open("data/bestword2vec.pickle", "wb") as outputfile:
        pickle.dump(tmpModel, outputfile)
    with open("data/bestdoc2vec.pickle", "wb") as outputfile:
        pickle.dump(tmpModel2, outputfile)


def uclaDataSaver():
    try:
        with open("data/bestword2vec.pickle", "rb") as fp:
            tmpModel = pickle.load(fp)
        with open("data/bestdoc2vec.pickle", "rb") as fp:
            tmpModel2 = pickle.load(fp)
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
        vec3 = tmpModel2.vecModelEstimator.model.infer_vector(
            p1.process(words))
        ucladoc2vecJson[cid] = list(map(lambda x: str(x), vec3))
        vecs1, vecs2 = [], []
        for word in p1.process(words):
            try:
                vecs1.append(tmpModel.vecModelEstimator.model[word])
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
    def __init__(self, model_algo="word2vec"):
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
                    return

    def queryPrereqsCIDs(self, cids):
        '''
        find the number of the prereqsites courses(cids)
        input is a list of cids
        that are in the course(cid)'s top-3 related courses
        '''
        with open("data/courses.json", "r") as fp:
            COURSES = json.load(fp)
        res = []
        for cid in cids:
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
        '''
        query 
        '''
        with open("data/courses.json", "r") as fp:
            COURSES = json.load(fp)
        res = []
        for cid in cids:
            school = COURSES[cid]["school"]
            top10ID = self._queryCID(COURSES, cid, 10)
            print("Top10 related courses of "+cid + ": ", top10ID)
            ratio = np.mean([COURSES[rel_cid]["school"]
                             == school for rel_cid in top10ID])
            res.append(ratio)
        return res

    def _queryCID(self, COURSES, cid, k):
        '''
        find the prereq courses of the course(cid) 
        '''

        top = self._queryTopKCID(cid=cid, k=k, COURSES=COURSES)
        topID = [c[0] for c in top]
        return topID

    def _queryTopKCID(self, cid, k, COURSES):
        '''
        return the top k related courses of a course.
        k should be smaller than or equal to 10 in this case
        '''
        assert(k <= 10)
        if self.model.algo == "Levenshtein":
            string = COURSES[cid]["name"]
        else:
            string = COURSES[cid]["desc"]
        return self.stringQuery(string)[:k]

    def queryUCLACID(self, cid):
        '''
        Based on the link provided, only courses from the computer science department are considered.

        This function has logic overlap with queryString() but queryString() only allows quering on usc courses.
        '''
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
                score = np.dot(uscVec,uclaVec) / \
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
            uscVec = [float(x) for x in  uscDoc2Vec[cid]]
            resScore, resId, resName = -1, None, None  # magic number
            for uclaID, uclaVec in uclaDoc2Vec.items():
                uclaVec = [float(x) for x in uclaVec]
                score = np.dot(uscVec,uclaVec)/np.linalg.norm(uclaVec)/np.linalg.norm(uscVec)
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
    interNum = len(list((Counter(ls1) & Counter(ls2)).elements()))
    return interNum/(len(ls1)+len(ls2)-interNum)
