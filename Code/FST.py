import syllabliser
import random

mefulumefailu = ["-", "-", ".", ".", "-", "-", ".", ".", "-", "-", ".", ".", "-", "-"]
failatunfailautun = ["-", ".", "-", "-", "-", ".", "-", "-", "-", ".", "-", "-", "-", ".", "-"]

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

    def __init__(self, metre, wordsPath, errTh=0):
        self.errThreshold = errTh
        self.metre = metre
        
        # Read vocabulary
        f = open(wordsPath, 'r')
        plainWords = f.readlines()
        f.close()
        
        # Create vocabulary
        self.vocabulary = list()
        for x in plainWords:
            x = x.strip()
            self.vocabulary.append((x, syllabliser.get_aruz(x)))
        self.vocabulary.append(("<endLine>", []))
        endLine = len(self.vocabulary) - 1
        self.vocabulary.append(("<endCouplet>", []))
        endCouplet = len(self.vocabulary) - 1

        self.lineLength = len(metre)
        self.rootMachine = [[] for i in range(self.lineLength + 1)]
        self.rootMachine[self.lineLength].append((endLine, 1))

        # For each word in the vocabulary
        for i in range(0, len(self.vocabulary) - 2):  # Do not iterate over the <endCouplet> and <endLine> tokens
            (word, met) = self.vocabulary[i]
            # Temporarily use -. for lines ending with a long vowel followed by a consonant
            met = syllabliser.elongateMetre(met)

            # For each state of the root machine
            for s in range(0, len(self.rootMachine) - 1):
                # If you can use the word at state s without crossing the ending of the metre
                if not s + len(met) > len(self.metre):
                    # If the word fits the rhythmic metre within a given error
                    if not computeErr(self.metre[s:s + len(met)], met) > self.errThreshold:
                        # Add the word to the machine as (word, distance to nextState)
                        self.rootMachine[s].append((i, len(met)))
        # Copy and concatenate a copy of the machine to the end
        self.rootMachine += self.rootMachine
        # Finish the machine for a couplet
        self.rootMachine.append([(endCouplet, 0)])
        # Update internal variables for number of states and the machine
        self.reset()

    def save(self):
        pass

    def reset(self):
        self.machine = self.rootMachine
        self.states = len(self.machine)

    def generate(self):
        state = 0
        strr = ""
        while state < fst.states:
            r = random.randint(0, len(fst.machine[state]) - 1)
            word, step = fst.machine[state][r]
            strr += fst.vocabulary[word][0] + " "
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
        for i in range(0, len(self.vocabulary)):
            if self.vocabulary[i][0] == word:
                return (i, self.vocabulary[i][1])
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


vezn = failatunx4
fst = FST(vezn, "./revani-words")
for i in range(0,10):
    print(fst.generate().replace("<endLine>","\n").replace("<endCouplet>","\n-------\n"))
#fst.constrain(1, 'ḥüsndür bir ḥüsninüñ')
