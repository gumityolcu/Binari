import syllabliser
import random
import model
import copy

mefulumefailu = ["-", "-", ".", ".", "-", "-", ".", ".", "-", "-", ".", ".", "-", "-"]
failatunfailatun = ["-", ".", "-", "-", "-", ".", "-", "-", "-", ".", "-", "-", "-", ".", "-"]
failatun = ["-", ".", "-", "-"]
feilatun = [".", ".", "-", "-"]
mefailun = [".","-","-","-"]
mefailunmefailun=[".","-","-","-",".","-","-","-",".","-","-","-",".","-","-","-"]

def computeErr(arr1, arr2, EOL=False):
    err = 0
    MAX = 10000
    for i in range(0, len(arr1)):
        if arr1[i] != arr2[i]:
            if not (i==len(arr1)-1 and EOL): # If the mistake is not at the end of the line
                if arr1[i] == "-" and arr2[i] == ".":
                    err = err + 1
                else:
                    return MAX
    return err

def makeWordList(inp,outp):
    f=open(inp,"r")
    lines=f.readlines()
    f.close()
    words=list()
    for l in lines:
        spl=l.split()
        for w in spl:
            words.append(w)
    f=open(outp,"w")
    words=sorted(list(set(words)))
    for i,w in enumerate(words):
        if w!="<beginCouplet>" and w!="<endCouplet>" and w!="<endLine>":
            w = w.lower()
            w = w.replace("s̲", "S")
            f.write(w)
            if i!=len(words)-1:
                f.write("\n")
    f.close()


class FST:

    def __init__(self, machinePath):
        # Load from already existing file
        pass

    def __str__(self):
        ret=""
        for i in range(0, self.states):
            ret+=str(i) + " " + str(self.machine[i])+"\n"
        return ret

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
            y=x
            x = x.strip()
            self.vocabulary.append((x, syllabliser.get_aruz(x)))
        self.vocabulary.append(("<beginCouplet>",[]))
        self.beginCouplet = len(self.vocabulary) - 1
        self.vocabulary.append(("<endLine>", []))
        self.endLine = len(self.vocabulary) - 1
        self.vocabulary.append(("<endCouplet>", []))
        self.endCouplet = len(self.vocabulary) - 1

        self.lineLength = len(metre)
        self.rootMachine = [[] for i in range(self.lineLength + 1)]
        self.rootMachine[self.lineLength].append((self.endLine, 1))

        # For each word in the vocabulary
        for i in range(0, len(self.vocabulary) - 3):  # Do not iterate over the <beginCouplet>, <endCouplet> and <endLine> tokens
            (word, met) = self.vocabulary[i]
            # Temporarily use -. for lines ending with a long vowel followed by a consonant
            met = syllabliser.elongateMetre(met)

            # For each state of the root machine
            for s in range(0, len(self.rootMachine) - 1):
                if word=="<izafe>":
                    if (s < len(self.rootMachine) - 2) and (s>0):
                    # add <izafe> as usable for all syllables except the last syllable of a line or the first syllable of a line
                        self.rootMachine[s].append((i,1))
                # If you can use the word at state s without crossing the ending of the metre
                elif not s + len(met) > len(self.metre):
                    # If the word fits the rhythmic metre within a given error
                    # (accounting for the fact that the last syllable is allowed
                    # to be any kind of syllable)
                    if not computeErr(self.metre[s:s + len(met)], met, (s+len(met)==len(self.metre))) > self.errThreshold:
                        # Add the word to the machine as (word, distance to nextState)
                        self.rootMachine[s].append((i, len(met)))
        # Copy and concatenate a copy of the machine to the end
        self.rootMachine += copy.deepcopy(self.rootMachine)
        # Finish the machine for a couplet
        self.rootMachine.append([(self.endCouplet, 0)])
        # Add the beginning state of the machine
        self.rootMachine.insert(0, [(self.beginCouplet,1)])
        # Update internal variables for number of states and the machine
        self.reset()

    def save(self):
        pass

    def reset(self):
        #temp=copy.deepcopy(self.rootMachine)
        self.machine = copy.deepcopy(self.rootMachine)
        self.states = len(self.machine)

    def findEdgesEndingIn(self,state):
        ret=[]
        for i,s in enumerate(self.machine):
            for e in s:
                if i+e[1]==state:
                    ret.append(e)
        return ret


    def reverse(self):
        tempMachine=copy.deepcopy(self.machine)
        stateCount=len(tempMachine)
        tempMachine[-1]=[]
        tempMachine[0]=[]
        newMachine=[[] for i in range(0, stateCount)]
        newMachine[0]=[(self.machine[-1][0][0],1)]
        newMachine[-1]=[(self.machine[0][0][0],0)]
        for s in range(1, stateCount-1):
            edges=self.findEdgesEndingIn(stateCount-s)
            newState=[]
            for e in edges:
                if e[0]!=self.endCouplet and e[0]!=self.beginCouplet:
                    newState.append(e)
            newMachine[s]=newState
        self.machine=newMachine
        return self.machine

    def generate(self, initial_state=0):
        state = initial_state
        strr = ""
        while state < self.states:
            r = random.randint(0, len(self.machine[state]) - 1)
            word, step = self.machine[state][r]
            strr += self.vocabulary[word][0] + " "
            state = state + step
            if step == 0:
                state = state + 1
        return strr

    # Format coupler
    def format(self, str):
        str = str.replace(" <endLine> ", "\n")
        str = str.replace("<endCouplet> ", "")
        return str

    # Generate and format output
    def formatted(self):
        return self.format(self.generate())

    def get_word_from_id(self, id):
        return self.vocabulary[id]

    def get_word_from_string(self, word):
        for i in range(0, len(self.vocabulary)):
            if self.vocabulary[i][0] == word:
                return (i, self.vocabulary[i][1])
        return -1

    def constrain(self, line, inWords):
        if inWords=="":
            return
        interval_init = 1 + line * (len(self.metre) + 1)
        interval_end = interval_init + len(self.metre)

        wordList = inWords.split(" ")
        wordList.reverse()
        getWords = []
        words = []
        for w in wordList:
            wrd = self.get_word_from_string(w)
            # If the w is found in the vocabulary
            if wrd != -1:
                getWords.append(wrd)
            else:
                raise ValueError("Word \"" + w +"\" not found in the vocabulary")
        for w in getWords:
            # w = (word_id, word_aruz)
            # Use elongated aruz patterns
            words.append((w[0], syllabliser.elongateMetre(w[1])))
        rhymeLength = 0
        for w in words:
            rhymeLength += len(w[1])

        # For each state in the constrained line's interval
        for s in range(interval_init, interval_end):
            if s + rhymeLength > interval_end:
                self.machine[s] = []
            # If a word in this state, *followed by the rhyme words* passes the end of the line
            w = 0
            while w < len(self.machine[s]):
                wrdLen = self.machine[s][w][1]
                if s + wrdLen + rhymeLength > interval_end:
                    self.machine[s].pop(w)
                else:
                    # Only increment w when no item has been deleted
                    w = w + 1
        # All wrong words are deleted, time to add the rhyme words to the machine
        curState = interval_end
        for w in words:
            length = len(w[1])
            curState = curState - length
            self.machine[curState].append((w[0], length))

if __name__=='__main__':
    outp = "./data/tempWordList.txt"
    makeWordList("./data/OTAP clean data/total", outp)
    # vezn = FST.mefailunmefailun
    vezn = failatunfailatun
    fst = FST(vezn, outp)
    constraint1 = "ḫān mısın kāfir"
    constraint2 = "sūzān mısın kāfir"
    fst.constrain(0, constraint1)
    fst.constrain(1, constraint2)
    #print(str(fst))
    print(fst.generate())
    print("\n**********************\n")
    fst.reverse()
    print(fst.generate())
    fst.reverse()
    print(fst.generate())
    print(str(fst))











    #outp="./data/OTAP clean data/wordList.txt"
    #makeWordList("./data/OTAP clean data/total-transcription",outp)
    #vezn = failatun
    #fst = FST(vezn, outp)


    #fst.constrain(0, 'āşiyān eyler beni')
    #fst.constrain(1, "revān eyler beni")
    #print("\n*************\n")
    #print(fst)