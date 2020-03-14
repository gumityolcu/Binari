import syllabliser
import random


def computeErr(arr1, arr2):
    err = 0
    MAX = 10000
    for i in range(0, len(arr1)):
        if arr1[i] != arr2[i]:
            if arr1[i] == "-" and arr2[i] == ".":
                err = err + 1
            else:
                return MAX
    return err


class FST:

    def __init__(self, machinePath):
        # Load from already existing file
        pass

    def __init__(self, metre, wordsPath):
        self.errThreshold = 0
        self.metre = metre
        f = open(wordsPath, 'r')
        plainWords = f.readlines()
        f.close()
        self.words = list()
        for x in plainWords:
            x = x.strip()
            self.words.append((x, syllabliser.get_aruz(x)))
        self.words.append(("<endLine>", []))
        endLine = len(self.words) - 1
        self.words.append(("<endCouplet>", []))
        endCouplet = len(self.words) - 1

        self.lineLength = len(metre)
        self.rootMachine = [[] for i in range(self.lineLength + 1)]
        self.rootMachine[self.lineLength].append((endLine, 1))

        for i in range(0, len(self.words) - 2):  # Do not iterate over the <endCouplet> and <endLine> tokens
            (wordd, met) = self.words[i]
            longMetre = syllabliser.elongateMetre(met)

            for s in range(0, len(self.rootMachine) - 1):
                if not s + len(longMetre) > len(self.metre):
                    if not computeErr(self.metre[s:s + len(longMetre)], longMetre) > self.errThreshold:
                        self.rootMachine[s].append((i, len(longMetre)))
        self.rootMachine += self.rootMachine
        self.rootMachine.append([(endCouplet, 0)])
        self.states = len(self.rootMachine)
        self.machine = self.rootMachine

    def reset(self):
        self.machine = self.rootMachine

    def generate(self):
        state = 0
        strr = ""
        while state < fst.states:
            r = random.randint(0, len(fst.machine[state]) - 1)
            word, step = fst.machine[state][r]
            strr += fst.words[word][0] + " "
            state = state + step
            if step == 0:
                state = state + 1
        return strr

    def format(self, str):
        str = str.replace(" <endLine> ", "\n")
        str = str.replace("<endCouplet> ", "")
        return str

    def formatted(self):
        return self.format(self.generate())

    def get_word(self, word):
        for i in range(0, len(self.words)):
            if self.words[i][0] == word:
                return (i, self.words[i][1])
        return -1

    def constrain(self, line, inWords):
        intervals = [(0, len(self.metre)), (len(self.metre)+1, 2*(len(self.metre))+1)]
        interval = intervals[line]
        if line == 1:
            offset = len(self.metre)
        wordList = inWords.split(" ")
        getWords = []
        words = []
        for w in wordList:
            wrd = self.get_word(w)
            if wrd != -1:
                getWords.append(wrd)
        for w in getWords:
            words.append((w[0], syllabliser.elongateMetre(w[1])))
        countSyl=0
        for w in words:
            countSyl+=len(w[1])
        interval=(interval[0], interval[1]-countSyl)
        print(interval)


mefulumefailu = ["-", "-", ".", ".", "-", "-", ".", ".", "-", "-", ".", ".", "-", "-"]
failatunx4 = ["-", ".", "-", "-", "-", ".", "-", "-", "-", ".", "-", "-", "-", ".", "-"]
vezn = failatunx4
fst = FST(vezn, "./revani-words")
fst.constrain(1, 'ḥüsndür bir ḥüsninüñ')
