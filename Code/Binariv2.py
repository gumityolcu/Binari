# THIS BINARY USES SYLLABLE LEVEL FSM TO SELECT TOKENS

import tensorflow as tf
import FST
import syllabliser as syl
import model
import random
import numpy as np
import sys
import time
import copy

sys.setrecursionlimit(20000)


# Given unnormalized log probabilities, get the probability of a given index
def get_probability(logits, index):
    denom = np.sum(np.exp(logits))
    num = np.exp(logits[index])
    return num / denom


# Given a hidden state of GRU and a list of token indices,
# returns the score given to those tokens and the new hidden state after processing those tokens
def getScore(mdl, h,
             tokens, normalize=True):
    mdl.layers[1].reset_states(states=h)
    score = 0.0
    for t in tokens:
        # Get logits from final fully connected layer with model.layers[2](h) and get the probability of the given token
        hSt = mdl.layers[1].states[0]
        logits = mdl.layers[2](hSt)
        p = tf.squeeze(tf.nn.softmax(logits), 0)[t]
        score += np.log(p)
        mdl(tf.expand_dims(np.array([t]), 0))  # Update model hidden state
    if normalize:
        score /= len(tokens)
    return mdl.layers[1].states[0].numpy(), score


def makeIdxfromChar(str, char2idx, level):
    tokens = []
    if level == "CHAR":
        tokens = syl.get_chars(str)
    elif level == "SYL":
        tokens = syl.get_syllables(str)
    return [char2idx[t] for t in tokens]


# Beam state is (generated-text, fst-state, hidden-state, score)
def expandBeam(beam, fst, mdl, char2idx, level, BACKWARD=True):
    # print("Expanding: "+beam[0])
    generated_text = beam[0]
    fst_state = beam[1]
    hidden_state = beam[2]
    score = beam[3]
    retBeams = []
    if fst_state == fst.states - 1:
        # If already printed the last token
        if generated_text[-1][-8:] == "Couplet>":
            return [beam]
    lastLength = 0
    FSTWords = copy.deepcopy(fst.machine[fst_state])
    FSTWords.append((char2idx[" "], 0))
    for i, s in enumerate(fst.machine[fst_state]):  # For each next state
        if len(fst.machine[fst_state]) != 1:
            if i % 100 == 0:
                printMsg = "Expanding: " + str(i * 100 / len(fst.machine[fst_state]))
                print(printMsg)
        word, step = s
        if word != char2idx[" "]:
            textWord = fst.vocabulary[word][0]
        else:
            textWord = " "
        # update generated-text
        newText = generated_text + [textWord]
        # update fst-state
        newState = fst_state + step

        # update hidden-state and score
        # mdl.layers[1].reset_states(states=hidden_state)
        indexes = makeIdxfromChar(fst.vocabulary[word][0], char2idx, level)
        if BACKWARD:
            indexes.reverse()
        newH, newScore = getScore(mdl, hidden_state, indexes)

        # Continue adding tokens recursively to the beam if there is only one choice for the next state
        # Return this further expanded beam or beam with possible next expantions
        if len(fst.machine[newState]) == 1:
            nextWord, _ = fst.machine[newState][0]
            newBeam = (newText, newState, newH, score + newScore)
            retBeams += expandBeam(newBeam, fst, mdl, char2idx, level)
        else:
            newBeam = (newText, newState, newH, score + newScore)
            retBeams.append(newBeam)
    return retBeams


def generate_text(model, char2idx, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 300

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = start_string

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()

    tf.random.set_seed(42)
    for i in range(num_generate):
        # if i > 200:
        #    print(str(i) + text_generated[-1])
        predictions = model(input_eval)
        # remove the batch
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions[:, 1:], num_samples=1)[-1, 0].numpy() + 1

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return text_generated


def getStringFromArray(arr):
    ret = ""
    for w in arr:
        ret += w
    return ret


def beamArrayToStr(bArray):
    ret = "\n"
    for b in bArray:
        ret += getStringFromArray(b[0]) + "\n"
    return ret


def selectBeams(beams, size, randomize=False):
    if len(beams) < size:
        return beams
    else:
        temp=[]
        if len(beams)<2*size:
            temp=sorted(beams, key=lambda b: b[3], reverse=True)
        else:
            temp=sorted(beams, key=lambda b: b[3], reverse=True)[0:size*2]

        if randomize:
            ret=random.sample(temp,size)
        else:
            ret = temp[0:size]
        return ret


def getLegalSyllables(char2idx, metres):
    ret = []
    for s in char2idx.keys():
        if s!="<beginCouplet>" and s!="<endLine>" and s!="<endCouplet>" and s!="<izafe>" and s!=" " and s!="<PAD>":
            syla = syl.get_aruz(s)
            for met in metres:
                if syla == met:
                    ret.append(s)
    return ret


def generateLine(metre, langModel, char2idx, beams, line, BEAMSIZE, randomize=False):
    cont = True
    if line==1:
        lineEnd="<endLine>"
    else:
        lineEnd="<beginCouplet>"
    while (cont):
        cont = False
        newBeams = []
        print(str(len(beams)) + " beams")
        for b in beams:
            generated_text = b[0]
            line_state = b[1]
            hidden_state = b[2]
            score = b[3]
            if line_state<len(metre):
                print("Expanding: '" + getStringFromArray(generated_text) + "'")
                # BEAM EXPANSION
                # Get legal syllable type
                legalMetres = []
                if metre[line_state] == "-":
                    if line_state == 0: # Last syllable in the non-reversed metre can be an open syllable
                        legalMetres.append(["."])
                    legalMetres.append(["-"])
                    if line_state < len(metre) - 3:
                        if metre[line_state + 1] == "-" and metre[line_state + 2] == ".":
                            legalMetres.append([".", "-", "-"])  # Add <mahlas>
                else:
                    if line_state<len(metre)-1:
                        if metre[line_state + 1] == "-":
                            legalMetres.append(["-."])
                    legalMetres.append(["."])

                legalSyllables = getLegalSyllables(char2idx,legalMetres)
                # Handle <izafe> and " "
                lastToken=generated_text[-1]
                if line_state!=0 and line_state!=len(metre)-1:
                    legalSyllables.append("<izafe>")
                if lastToken!=" ":
                    legalSyllables.append(" ")
                # Expand beam with these syllables
                freshBeams=[]
                for i,s in enumerate(legalSyllables):
                    if i % 100 == 0:
                        printMsg = "Expanding: " + str(i * 100 / len(legalSyllables))
                        print(printMsg)
                    newH, sc = getScore(langModel,hidden_state,[char2idx[s]])
                    newText=generated_text+[s]
                    newState=line_state
                    if s!=" ":
                        newState+=len(syl.elongateMetre(syl.get_aruz(s)))
                    if not newState<len(metre):
                        newnH, newsc=getScore(langModel, newH, [char2idx[lineEnd]])
                        newText+=[lineEnd]
                        sc+=newsc
                    tempBeam=(newText, newState, newH, score+sc)
                    freshBeams.append(tempBeam)
                print("\n", end="")
                print("Received " + str(len(freshBeams)) + " beams")
                newBeams += freshBeams
                for b in newBeams:
                    if b[1]<len(metre):
                        cont = True
            else:
                newBeams.append(b)
        beams = selectBeams(newBeams, BEAMSIZE,randomize=randomize)
        print("Selected beams:" + beamArrayToStr(beams))
    return beams


def generateCouplet(metre, langModel, char2idx, BEAMSIZE=5, constraint1="", constraint1Count=0, constraint2="",
                    constraint2Count=0, randomize=False):
    metre.reverse()
    stateShape = langModel.layers[1].states[0].numpy().shape
    s_0 = np.zeros(shape=stateShape)
    langModel.layers[1].reset_states(states=s_0)
    init = []
    init.append("<endCouplet>")
    init.append("<endLine>")
    spl = constraint2.split()
    spl.reverse()
    count = 0
    for s in spl:
        if s == "_":
            s = " "
        init.append(s)
    indexes=[]
    for i in init:
        indexes.append(char2idx[i])
    h, sc = getScore(langModel, s_0, indexes)

    beam = (init, constraint2Count, h, sc)
    beams = []
    beams.append(beam)
    newBeams = generateLine(metre, langModel, char2idx, beams, line=1, BEAMSIZE=BEAMSIZE,randomize=randomize)
    spl = constraint1.split()
    init = []
    spl.reverse()
    for s in spl:
        if s == "_":
            s = " "
        init.append(s)
    indexes=[]
    for i in init:
        indexes.append(char2idx[i])
    for i in range(len(newBeams)):
        h, sc = getScore(langModel, newBeams[i][2], indexes)
        newBeams[i] = (newBeams[i][0] + init, constraint1Count, h, newBeams[i][3] + sc)

    newBeams = generateLine(metre, langModel, char2idx, newBeams, line=2, BEAMSIZE=BEAMSIZE,randomize=randomize)
    return newBeams


if __name__ == "__main__":
    start = time.time()
    LEVEL = sys.argv[1]
    BACKWARD = True
    EP = sys.argv[2]
    BEAMSIZE = sys.argv[3]
    resultPath="./Code/Experiments/Real Experiments/"+sys.argv[4]+"/results"
    RANDOMIZE = False
    DATA = "real"  # toy or real
    random.seed(42)
    checkpoint_path = "Code/Checkpoints/"
    _, syllables, _ = model.createSylLevelData("data/OTAP clean data/total")
    if LEVEL == "SYL":
        _, idx2char, char2idx = model.createSylLevelData("data/OTAP clean data/total")
        checkpoint_path += "SyllableLevel/Transcription/"
    #elif LEVEL == "CHAR":
    #    checkpoint_path += "CharacterLevel/Transcription/"
    #    _, idx2char, char2idx = model.createCharLevelData("data/OTAP clean data/total", pad=True)
    else:
        print("Invalid LEVEL parameter")
        exit()
    if BACKWARD:
        checkpoint_path += "Backward/"
    else:
        checkpoint_path += "Forward/"
    checkpoint_path += str(EP) + "/ckpt" + str(EP)
    langModel = model.buildGRUModelWithEmbedding(vocab_size=len(idx2char), embedding_dim=128, rnn_units=256,
                                                 batch_size=1,
                                                 statefulness=True)
    print(checkpoint_path)

    langModel.load_weights(checkpoint_path)
    langModel.build(tf.TensorShape([1, None]))
    langModel.summary()
    print("Text generated by model: ")
    if BACKWARD:
        reverseText = generate_text(langModel, char2idx, ['<endCouplet>', '<endLine>'])
        reverseText.reverse()
        print(''.join(reverseText))
    else:
        print(''.join(generate_text(langModel, char2idx, ['<beginCouplet>'])))

        # vezn = FST.mefailunmefailun
    vezn = FST.mefailunmefailun
    f = open("data/OTAP clean data/syllables", "w")
    for i, s in enumerate(syllables):
        if s != " " and s != "<PAD>" and s != "<beginCouplet>" and s != "<endLine>" and s != "<endCouplet>":
            f.write(s)
            if i != len(syllables) - 1:
                f.write("\n")
    f.close()
    constraint1 = "_ cān"
    constraint1Count=1
    constraint2 = "_ sū zān"
    constraint2Count=2
    beyts = generateCouplet(vezn, langModel, char2idx, BEAMSIZE=BEAMSIZE, constraint1=constraint1, constraint1Count=constraint1Count, constraint2=constraint2, constraint2Count=constraint2Count, randomize=RANDOMIZE)
    end = time.time()
    f = open(resultPath, "w")
    f.write("Beam size: " + str(BEAMSIZE) + "\n")
    f.write("Constraints:\n")
    f.write("Constraint on line 1: ")
    if constraint1 == "":
        f.write("None")
    else:
        for c in constraint1.split():
            f.write(c.replace("_"," "))
    f.write("\n")
    f.write("Constraint on line 2: ")
    if constraint2 == "":
        f.write("None")
    else:
        for c in constraint2.split():
            f.write(c.replace("_", " "))
    f.write("\n")
    for b, _, _, sc in beyts:
        if BACKWARD:
            b.reverse()
        print(str(b) + " " + str(sc))
        for x in b:
            f.write(x)
        f.write("\t")
        f.write(str(sc))
        f.write("\n")
    f.close()
    print("Time elapsed in seconds : " + str(start - end))
    print("Time elapsed in minutes : " + str((start - end) * 1.0 / 60))
    print("Time elapsed in hours : " + str((start - end) * 1.0 / 3600))

"""

# Word FSM main
if __name__ == "__main__":
    start=time.time()
    LEVEL = "SYL"
    BACKWARD=True
    EP=1000
    BEAMSIZE=5
    DATA="real" #toy or real
    checkpoint_path = "Code/Checkpoints/"
    if LEVEL == "SYL":
        _, idx2char, char2idx = model.createSylLevelData("data/OTAP clean data/total")
        checkpoint_path += "SyllableLevel/Transcription/"
    elif LEVEL == "CHAR":
        checkpoint_path += "CharacterLevel/Transcription/"
        _, idx2char, char2idx = model.createCharLevelData("data/OTAP clean data/total", pad=True)
    else:
        print("Invalid LEVEL parameter")
        exit()
    if BACKWARD:
        checkpoint_path+="Backward/"
    else:
        checkpoint_path+="Forward/"
    checkpoint_path+=str(EP)+"/ckpt"+str(EP)
    langModel = model.buildGRUModelWithEmbedding(vocab_size=len(idx2char), embedding_dim=128, rnn_units=256,
                                                 batch_size=1,
                                                 statefulness=True)
    print(checkpoint_path)

    langModel.load_weights(checkpoint_path)
    langModel.build(tf.TensorShape([1, None]))
    langModel.summary()
    print("Text generated by model: ")
    if BACKWARD:
        reverseText=generate_text(langModel,char2idx,['<endCouplet>','<endLine>'])
        reverseText.reverse()
        print(''.join(reverseText))
    else:
        print(''.join(generate_text(langModel,char2idx,['<beginCouplet>'])))


    if DATA=="real":
        outp = "./data/OTAP clean data/wordList.txt"
    else:
        outp = "./data/tempWordList.txt"
    makeSyllableList("./data/OTAP clean data/syllables")
    #vezn = FST.mefailunmefailun
    vezn=FST.mefailunmefailun
    fst = FST.FST(vezn, outp)
    constraint1 = "ḫān mısın kāfir"
    constraint2 = "sūzān mısın kāfir"
    fst.constrain(0,constraint1)
    fst.constrain(1,constraint2)
    if BACKWARD:
        fst.reverse()
    beyts=generateCouplet(fst,langModel,BEAMSIZE=BEAMSIZE, BACKWARD=BACKWARD)
    end=time.time()
    f=open("./Code/Experiments/Exp5(SyllableGRU-1000-Backward-WordFSM-Normalized)/results","w")
    f.write("Beam size: "+str(BEAMSIZE)+"\n")
    f.write("Constraints:\n")
    f.write("Constraint on line 1: ")
    if constraint1=="":
        f.write("None")
    else:
        f.write(constraint1)
    f.write("\n")
    f.write("Constraint on line 2: ")
    if constraint2=="":
        f.write("None")
    else:
        f.write(constraint2)
    f.write("\n")
    for b,_,_,sc in beyts:
        if BACKWARD:
            b.reverse()
        print(str(b)+" "+str(sc))
        for x in b:
            f.write(x)
        f.write("\t")
        f.write(str(sc))
        f.write("\n")
    f.close()
    print("Time elapsed in seconds : "+str(start-end))
    print("Time elapsed in minutes : "+str((start-end)*1.0/60))
    print("Time elapsed in hours : "+str((start-end)*1.0/3600))





"""
