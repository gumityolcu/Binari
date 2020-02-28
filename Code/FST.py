import syllabliser
import random

def computeErr(arr1, arr2):
    err = 0
    for i in range(0, len(arr1)):
        if arr1[i] != arr2[i]:
            err = err + 1
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

        self.lineLength = len(metre)
        self.rootStates = [[] for i in range(self.lineLength+1)]

        for i in range(0, len(self.words)):
            (wordd, met) = self.words[i]
            longMetre = []
            for a in met:
                if a == "-.":
                    longMetre.append("-")
                    longMetre.append(".")
                else:
                    longMetre.append(a)

            for s in range(0, len(self.rootStates) - 1):
                if not s+len(longMetre)>len(self.metre):
                    if not computeErr(self.metre[s:s+len(longMetre)], longMetre)>self.errThreshold:
                        self.rootStates[s].append((i, len(longMetre)))


vezn=["-",".","-","-","-",".","-","-","-",".","-","-","-",".","-"]
fst = FST(vezn, "./dummy")
state=0
strr=""
while state<len(vezn):
    r=random.randint(0,len(fst.rootStates[state])-1)
    word,step=fst.rootStates[state][r]
    strr+=fst.words[word][0]+" "
    state=state+step
print(strr)
