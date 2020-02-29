import syllabliser
import random

def computeErr(arr1, arr2):
    err = 0
    MAX=10000
    for i in range(0, len(arr1)):
        if arr1[i]!=arr2[i]:
            if arr1[i]=="-" and arr2[i]==".":
                err = err+1
            else:
                return MAX
    return err

class FST:

    def __init__(self, machinePath):
        # Load from already existing file
        pass

    def generate(self):
        state = 0
        strr = ""
        while state < fst.states:
            r = random.randint(0, len(fst.machine[state]) - 1)
            word, step = fst.machine[state][r]
            strr += fst.words[word][0]+" "
            state = state + step
            if step == 0:
                state = state + 1
        return strr

    def format(self, str):
        str=str.replace(" <endLine> ","\n")
        str=str.replace("<endCouplet> ","")
        return str

    def formatted(self):
        return self.format(self.generate())

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
        self.words.append(("<endLine>",[]))
        endLine=len(self.words)-1
        self.words.append(("<endCouplet>",[]))
        endCouplet=len(self.words)-1

        self.lineLength = len(metre)
        self.machine = [[] for i in range(self.lineLength+1)]
        self.machine[self.lineLength].append((endLine,1))

        for i in range(0, len(self.words)-2):#Do not iterate over the <endCouplet> and <endLine> tokens
            (wordd, met) = self.words[i]
            longMetre = []
            for a in met:
                if a == "-.":
                    longMetre.append("-")
                    longMetre.append(".")
                else:
                    longMetre.append(a)

            for s in range(0, len(self.machine) - 1):
                if not s+len(longMetre)>len(self.metre):
                    if not computeErr(self.metre[s:s+len(longMetre)], longMetre)>self.errThreshold:
                        self.machine[s].append((i, len(longMetre)))
        self.machine+=self.machine
        self.machine.append([(endCouplet,0)])
        self.states=len(self.machine)

mefulumefailu = ["-","-",".",".","-","-",".",".","-","-",".",".","-","-"]
failatunfailatun = ["-",".","-","-","-",".","-","-","-",".","-","-","-",".","-"]
vezn=failatunfailatun
fst = FST(vezn, "./revani-words")
for arse in range(0,15):
    print(fst.format(fst.generate()))