import bs4 as bs
import urllib.request
import re
import json
from tqdm import tqdm

def initialParse(url = "https://classes.usc.edu/term-20193/", 
                 school = "engineering"):
    source = urllib.request.urlopen(url)
    soup = bs.BeautifulSoup(source,"html.parser")
    if school == "engineering":
        return school, soup.findAll(attrs={"data-type" : "department",
                                   "data-school":"Engineering"})
    elif school == "medical":
        return school, soup.findAll(attrs={"data-type" : "department",
                                   "data-school":"Medicine"})
    else:
        raise Exception("wrong school argument: ",school)

def buildDepartments(school, departmentLists):
    Names = [a.get("data-title") for a in departmentLists]
    Hrefs = [next(a.children).get("href") for a in departmentLists]
    assert(len(Names)==len(Hrefs))
    return [Department(school, name,href) for name, href in zip(Names,Hrefs)]

class Department:
    def __init__(self,school, name,url):
        self.name = name
        self.url = url
        self.classLists = []
        self.school = school
        self.obtainClasses(school)
        
    def __repr__(self):
        return json.dumps({"department name": self.name,
                "department url": self.url})
    
    def findClassById(self,cid):
        for course in self.classLists:
            if course.cid == cid:
                return course
        return None
    def obtainClasses(self, school):
        source = urllib.request.urlopen(self.url)
        soup = bs.BeautifulSoup(source,"html.parser")
        courseLists = soup.findAll('div', {'class': re.compile(r'^course-info')})
        for course in courseLists:
            cid = course.get("id").replace("-","")
            url = course.find("a").get("href")
            name = course.find("a").text
            desc = course.find("div",{"class":"catalogue"}).text
            if(len(desc)) < 140: # MAGIC NUMBER
                continue
            # pre/corequisites
            notes = course.find("ul",{"class":"notes"}).findAll("li")
            prereqs = []
            coreqs = []
            if len(notes) != 0:
                for note in notes:
                    if note.get("class")[0]=="prereq":
                        for course in note.findAll("a"):
                            subCid = course.text.replace(" ","")
                            prereqs.append(subCid)
                    elif note.get("class")[0] =="coreq":
                        for course in note.findAll("a"):
                            subCid = course.text.replace(" ","")
                            coreqs.append(subCid)
            self.classLists.append(Class(cid,name,url,desc, prereqs, coreqs, school))   
                        
    
class Class:
    def __init__(self, cid, name, url, desc, prereqs, coreqs, school):
        self.cid = cid
        self.name = name
        self.url = url
        self.desc = desc
        self.prereqs = prereqs
        self.coreqs = coreqs
        self.school = school
    def __repr__(self):
        return json.dumps({
            "cid": self.cid,
            "course name": self.name,
            "course url": self.url,
            "course prereq": self.prereqs,
            "course coreqs": self.coreqs
        })