#Author: Carlos Eduardo L. Elmadjian
#-----
#this classifier requires BeautifulSoup package to work!

import sys
from bs4 import BeautifulSoup

STOP_WORDS = ["a", "about", "above", "above", "across", "after", "afterwards", \
              "again", "against", "all", "almost", "alone", "along", "already",\
              "also","although","always","am","among", "amongst", "amoungst",\
               "amount",  "an", "and", "another", "any","anyhow","anyone",\
               "anything","anyway", "anywhere", "are", "around", "as",  "at", \
               "back","be","became", "because","become","becomes", "becoming",\
               "been", "before", "beforehand", "behind", "being", "below",\
               "beside", "besides", "between", "beyond", "bill", "both",\
               "bottom","but", "by", "call", "can", "cannot", "cant", "co",\
               "con", "could", "couldnt", "cry", "de", "describe", "detail",\
               "do", "done", "down", "due", "during", "each", "eg", "eight",\
               "either", "eleven","else", "elsewhere", "empty", "enough",\
               "etc", "even", "ever", "every", "everyone", "everything",\
               "everywhere", "except", "few", "fifteen", "fify", "fill",\
               "find", "fire", "first", "five", "for", "former", "formerly",\
               "forty", "found", "four", "from", "front", "full", "further",\
               "get", "give", "go", "had", "has", "hasnt", "have", "he",\
               "hence", "her", "here", "hereafter", "hereby", "herein",\
               "hereupon", "hers", "herself", "him", "himself", "his", "how",\
               "however", "hundred", "ie", "if", "in", "inc", "indeed",\
               "interest", "into", "is", "it", "its", "itself", "keep", "last",\
               "latter", "latterly", "least", "less", "ltd", "made", "many",\
               "may", "me", "meanwhile", "might", "mill", "mine", "more",\
               "moreover", "most", "mostly", "move", "much", "must", "my",\
               "myself", "name", "namely", "neither", "never", "nevertheless",\
               "next", "nine", "no", "nobody", "none", "noone", "nor", "not",\
               "nothing", "now", "nowhere", "of", "off", "often", "on", "once",\
               "one", "only", "onto", "or", "other", "others", "otherwise",\
               "our", "ours", "ourselves", "out", "over", "own","part", "per",\
               "perhaps", "please", "put", "rather", "re", "same", "see",\
               "seem", "seemed", "seeming", "seems", "serious", "several",\
               "she", "should", "show", "side", "since", "sincere", "six",\
               "sixty", "so", "some", "somehow", "someone", "something",\
               "sometime", "sometimes", "somewhere", "still", "such",\
               "system", "take", "ten", "than", "that", "the", "their",\
               "them", "themselves", "then", "thence", "there", "thereafter",\
               "thereby", "therefore", "therein", "thereupon", "these", "they",\
               "thickv", "thin", "third", "this", "those", "though", "three",\
               "through", "throughout", "thru", "thus", "to", "together",\
               "too", "top", "toward", "towards", "twelve", "twenty", "two",\
               "un", "under", "until", "up", "upon", "us", "very", "via",\
               "was", "we", "well", "were", "what", "whatever", "when",\
               "whence", "whenever", "where", "whereafter", "whereas",\
               "whereby", "wherein", "whereupon", "wherever", "whether",\
               "which", "while", "whither", "who", "whoever", "whole", "whom",\
               "whose", "why", "will", "with", "within", "without", "would",\
               "yet", "you", "your", "yours", "yourself", "yourselves", "the",\
               "Reuter\n&#3;"]


#A processor capable of parsing sgml texts
##########################################
class SGMLProcessor():
    def __init__(self, file_descriptor):
        self.process_file(file_descriptor)

    def process_file(self, file_descriptor):
        soup = BeautifulSoup(file_descriptor, "xml")
        for r in soup.find_all('REUTERS'):
            print(r['LEWISSPLIT'], r['TOPICS'])
            print(r.BODY.string)



    def get_classes(self):
        pass

    def get_body_text(self):
        pass


#A Naive Bayes learner and classifier
#for the Reuters-21578 dataset
######################################
class NaiveBayes():
    def __init__(self):
        self.file = None

    def learn(self, training_set, classes):
        pass

    def classify(self, testing_set):
        pass

    def extract_vocabulary(self, training_set):
        pass

    def read_file(self, filename):
        with open(filename, "r") as f:
            processor = SGMLProcessor(f)


#main program
#############
def main():
    nb = NaiveBayes()
    nb.read_file(sys.argv[1])


if __name__ == "__main__":
    main()
