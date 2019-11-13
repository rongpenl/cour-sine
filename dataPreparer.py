from preprocessor import Preprocessor
import random
import numpy as np
import json
from collections import OrderedDict
from crawler import *


'''
an interface of data preparation.
function to learn:

a pair of alphabetically ordered courses as a string, separated by hepen:-> similarity score between 0 and 1.

training on descriptions are therefore hidden from the user.

'''


def splitDataSets(IDs, trainSize=0.7, seed=2019):
    '''
    Split a list of ids, can be uid or cid
    '''
    random.seed(seed)
    random.shuffle(IDs)
    cut = int(len(IDs)*trainSize)
    return IDs[:cut], IDs[cut:]


def generateUIDs(cids):
    '''
    build a list of UIDs from a list of cids
    '''
    uids = []
    for i in range(len(cids)):
        for j in range(i, len(cids)):
            ij = cids[i] < cids[j]
            # to ensure the uniqueness of uid
            if ij:
                uids.append(cids[i]+"-"+cids[j])
            else:
                uids.append(cids[j]+"-"+cids[i])
    return uids


def _generateGroundTruth(uids, COURSEDESCRIPTIONS):
    '''
    a list of uid with length n, return a n-by-3 dataframe
    the 2nd and 3rd columns are the raw string of course description
    '''
    gt = {}
    # _pseudoGroundTruth(COURSEDESCRIPTIONS)
    bertVecs = np.load("data/bert_vecs.npy")
    cidLists = list(COURSEDESCRIPTIONS.keys())
    for uid in uids:
        twoids = uid.split("-")
        id1, id2 = twoids[0], twoids[1]
        vec1, vec2 = bertVecs[cidLists.index(
            id1)], bertVecs[cidLists.index(id2)]
        sim = np.dot(vec1, vec2)/np.linalg.norm(vec1)/np.linalg.norm(vec2)
        # augment to [0,1]
        gt[uid] = sim
    # to ensure the similarities are comparable, 0-center the similarity value
    # ? variance TODO
    ave = np.mean(list(gt.values()))
    for key in gt.keys():
        gt[key] -= ave
    return gt


def _pseudoGroundTruth(COURSEDESCRIPTIONS):
    '''
    The psedu ground truth is obtained using the bert model
    Ron manually examined some results to ensure the quality

    https://github.com/hanxiao/bert-as-service
    https://towardsdatascience.com/
    nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b

    User shouldn't run this function. It is solely used to create the pseduo ground truth. 
    A bert-serving-server should run before calling this function

    '''
    from bert_serving.client import BertClient
    bc = BertClient()
    bertProc = Preprocessor()
    desc = list(map(lambda x: bertProc.stringNumRemover(bertProc.stringPunRemover(x.lower())),
                    list(COURSEDESCRIPTIONS.values())))
    encodedDesc = bc.encode(desc)
    # [CLS] field for classification
    vectors = [encodedDesc[i][0] for i in range(len(encodedDesc))]
    '''
    For t-SNE plotting:
    
    from sklearn.manifold import TSNE
    X_embedded = TSNE(n_components=2).fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(100,100))
    ax.scatter(X_embedded[:,0], X_embedded[:,1])
    for i, txt in enumerate(list(COURSEDESCRIPTIONS.keys())):
        ax.annotate(txt, (X_embedded[i,0], X_embedded[i,1]))
        
    '''
    return vectors


def _metaDataSave():
    '''
    A utility function to save all courses information to json file
    '''
    '''
    generate a course description json file in directory data
    '''
    school, departmentLists = initialParse(school="medical")
    medicalSchool = buildDepartments(school, departmentLists)
    school, departmentLists = initialParse(school="engineering")
    engineeringSchool = buildDepartments(school, departmentLists)
    COURSES = OrderedDict()

    # when train, build the data set on the fly
    for department in medicalSchool:
        for course in department.classLists:
            COURSES[course.cid] = {
                "name": course.name,
                "url": course.url,
                "prereqs": course.prereqs,
                "school": course.school,
                "desc": course.desc,
                "coreqs": course.coreqs
            }

    for department in engineeringSchool:
        for course in department.classLists:
            COURSES[course.cid] = {
                "name": course.name,
                "url": course.url,
                "prereqs": course.prereqs,
                "school": course.school,
                "desc": course.desc,
                "coreqs": course.coreqs
            }
    with open("data/courses.json","w") as fp:
        json.dump(COURSES,fp)
    # ucla
    uclaParse()


def initPrepare():
    '''
    generate a course description json file in directory data
    '''
    print("Begin parsing the websites")
    school, departmentLists = initialParse(school="medical")
    medicalSchool = buildDepartments(school, departmentLists)
    school, departmentLists = initialParse(school="engineering")
    engineeringSchool = buildDepartments(school, departmentLists)
    print("Begin building necessary data structures")
    cids = []
    desc = []
    COURSENAMES = OrderedDict()
    COURSEDESCRIPTIONS = OrderedDict()
    # when train, build the data set on the fly
    for department in medicalSchool:
        for course in department.classLists:
            cids.append(course.cid)
            desc.append(course.desc)
            COURSENAMES[course.cid] = course.name
            COURSEDESCRIPTIONS[course.cid] = course.desc

    for department in engineeringSchool:
        for course in department.classLists:
            cids.append(course.cid)
            desc.append(course.desc)
            COURSENAMES[course.cid] = course.name
            COURSEDESCRIPTIONS[course.cid] = course.desc

    # generate pseudo truth table
    uids = generateUIDs(cids)
    TRUTHTABLE = _generateGroundTruth(uids, COURSEDESCRIPTIONS)
    for key in TRUTHTABLE.keys():
        TRUTHTABLE[key] = str(TRUTHTABLE[key])
    for key in TRUTHTABLE.keys():
        TRUTHTABLE[key] = float(TRUTHTABLE[key])

    return cids, uids, COURSENAMES, COURSEDESCRIPTIONS, TRUTHTABLE
# build similarities data sets
