#Author: Carlos Eduardo L. Elmadjian
#-----
#this classifier requires BeautifulSoup package to work!

import re
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
               "reuter", "s"]


#A Naive Bayes learner and classifier
#for the Reuters-21578 dataset
######################################
class NaiveBayes():
    def __init__(self):
        self.vocabulary = {}
        self.V = {}
        self.prob_v = {}
        self.prob_w = {}
        self.docs = 0
        self.n = {}
        self.files = []

    def read_files(self, filenames):
        for filename in filenames:
            with open(filename, "r") as f:
                self.files.append(f.read())

    def learn(self):
        for f in self.files:
            self._process_file(f)
        self._calculate_probabilities()

    def _process_file(self, file_d):
        soup   = BeautifulSoup(file_d, "xml")
        for r in soup.find_all('REUTERS'):
            if r['LEWISSPLIT'] == 'TRAIN' and r.BODY and r.TOPICS.D:
                self.docs += 1
                topic = []

                #extracting topics
                for t in r.TOPICS:
                    topic.append(t.string)
                    if t.string not in self.V.keys():
                        self.V[t.string] = 0
                        self.n[t.string] = set()
                    self.V[t.string] = 1

                #extracting vocabulary
                words = re.findall('[A-Za-z]+', r.BODY.string)
                for w in words:
                    word = w.lower()
                    if word not in STOP_WORDS:
                        if word not in self.vocabulary.keys():
                            self.vocabulary[word] = {}
                        for t in topic:
                            if t not in self.vocabulary[word].keys():
                                self.vocabulary[word][t] = 0
                            self.vocabulary[word][t] += 1
                            self.n[t].add(word)

    def _calculate_probabilities(self):
        for w in self.vocabulary.keys():
            self.prob_w[w] = {}
            for v in self.V.keys():
                vsize = len(self.vocabulary.keys())
                nsize = len(self.n[v])
                self.prob_v[v] = self.V[v]/self.docs
                if v not in self.vocabulary[w].keys():
                    self.vocabulary[w][v] = 0
                self.prob_w[w][v] = (self.vocabulary[w][v] + 1)/(nsize + vsize)

    def classify(self, testing_set):
        """
        1.abrir o conjunto de arquivos marcados com LEWISSPLIT == 'TEST' e TOPICS == 'YES'
        2.para cada arquivo, testar o classificador e anotar o GT descrito na tag <TOPICS>
        3.comparar o resultado do teste de classificação para o argumento que maximiza com o GT
        4.mostrar a acurácia (em tabela --> porcentagem de acerto para cada classe)
        """
        pass






#main program
#############
def main():
    filenames =  ["reuters21578/reut2-00" + str(i) + ".sgm" for i in range(10)]
    filenames += ["reuters21578/reut2-0" + str(i) + ".sgm" for i in range(10,22)]
    nb = NaiveBayes()
    nb.read_files(filenames)
    nb.learn()


if __name__ == "__main__":
    main()
