import bs4 as bs
import urllib.request
import re
import json
from tqdm import tqdm


def initialParse(url="https://classes.usc.edu/term-20193/", school="engineering"):
    """the initialParse() parses the engineering/medical school websites and return the school name, as well as a raw list of html elements that contains department information. The natural downstream function is buildDepartments which takes those two values exactly.

    :param url: school url, defaults to "https://classes.usc.edu/term-20193/"
    :type url: str
    :param school: school, medical or engineering only, defaults to "engineering"
    :type school: str
    """
    source = urllib.request.urlopen(url)
    soup = bs.BeautifulSoup(source, "html.parser")
    if school == "engineering":
        departmentList = soup.findAll(attrs={"data-type": "department",
                                             "data-school": "Engineering"})
        return school, departmentList
    elif school == "medical":
        departmentList = soup.findAll(attrs={"data-type": "department",
                                             "data-school": "Medicine"})
        return school, departmentList
    else:
        print("wrong school argument: ", school)


def buildDepartments(school, departmentLists):
    """the downstream processing of initialParse(), returns a list of department objects. When initializing department, the course list will be built.
    
    :param school: school, engineering and medical only
    :type school: string
    :param departmentLists: a list of departments of given school.
    :type departmentLists: list
    :return: a list of department objects
    :rtype: list
    """
    Names = [a.get("data-title") for a in departmentLists]
    Hrefs = [next(a.children).get("href") for a in departmentLists]
    assert(len(Names) == len(Hrefs))
    return [Department(school, name, href) for name, href in zip(Names, Hrefs)]


def uclaParse():
    """The function will directly build a UCLA CS course webpage is special. Each course doesn't have its own url.
    """
    url = "https://www.registrar.ucla.edu/Academics/Course-Descriptions/Course-Details?SA=COM+SCI&funsel=3"
    source = urllib.request.urlopen(url)
    soup = bs.BeautifulSoup(source, "html.parser")
    courseList = soup.findAll(attrs={"class": "media-body"})
    uclaCourses = {}
    for idx, course in enumerate(courseList):
        cid = "COMSCI" + str(idx)
        name = course.find("h3").text
        desc = course.findAll("p")[1].text
        uclaCourses[cid] = {"cid": cid, "name": name, "url": None,
                            "desc": desc, "school": "engineering",
                            "prereqs": [], "coreqs": []}
    with open("data/uclacourses.json", "w") as fp:
        json.dump(uclaCourses, fp)


class Department:
    """The department class 
    """
    def __init__(self, school, name, url):
        """Constructor method
        
        :param school: school name
        :type school: string
        :param name: department name
        :type name: string
        :param url: url of department
        :type url: string
        """
        self.name = name
        self.url = url
        self.classLists = []
        self.school = school
        self.obtainClasses(school)

    def __repr__(self):
        return json.dumps({"department name": self.name,
                           "department url": self.url})

    def obtainClasses(self, school):
        """build the classLists field of the object. There is a design flaw passing 'school' around. Will fix it in future version.

        :param school: school name
        :type school: string
        """
        source = urllib.request.urlopen(self.url)
        soup = bs.BeautifulSoup(source, "html.parser")
        courseLists = soup.findAll(
            'div', {'class': re.compile(r'^course-info')})
        for course in courseLists:
            cid = course.get("id").replace("-", "")
            url = course.find("a").get("href")
            name = course.find("a").text
            desc = course.find("div", {"class": "catalogue"}).text
            if(len(desc)) < 140:  # MAGIC NUMBER
                continue
            # pre/corequisites
            notes = course.find("ul", {"class": "notes"}).findAll("li")
            prereqs = []
            coreqs = []
            if len(notes) != 0:
                for note in notes:
                    if note.get("class")[0] == "prereq":
                        for course in note.findAll("a"):
                            subCid = course.text.replace(" ", "")
                            prereqs.append(subCid)
                    elif note.get("class")[0] == "coreq":
                        for course in note.findAll("a"):
                            subCid = course.text.replace(" ", "")
                            coreqs.append(subCid)
            self.classLists.append(
                Class(cid, name, url, desc, prereqs, coreqs, school))


class Class:
    """The course(class) class.
    
    """
    def __init__(self, cid, name, url, desc, prereqs, coreqs, school):
        """Constructor function
        
        :param cid: course id
        :type cid: string
        :param name: course name
        :type name: string
        :param url: course url
        :type url: string
        :param desc: course description
        :type desc: string
        :param prereqs: list of prereqs
        :type prereqs: list of string
        :param coreqs: list of coreqs
        :type coreqs: list of string
        :param school: school of the department which offers the course
        :type school: string
        """
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
