import tensorflow as tf
import FST
import syllabliser as syl
import model
import numpy as np
import sys
import time

sys.setrecursionlimit(20000)


# Given unnormalized log probabilities, get the probability of a given index
def get_probability(logits, index):
    denom = np.sum(np.exp(logits))
    num = np.exp(logits[index])
    return num / denom


# Given a hidden state of GRU and a list of token indices,
# returns the score given to those tokens and the new hidden state after processing those tokens
def getScore(mdl, h,
             tokens):
    mdl.layers[1].reset_states(states=h)
    score = 0.0
    for t in tokens:
        # Get logits from final fully connected layer with model.layers[2](h) and get the probability of the given token
        hSt = mdl.layers[1].states[0]
        logits = mdl.layers[2](hSt)
        p = tf.squeeze(tf.nn.softmax(logits), 0)[t]
        score += np.log(p)
        mdl(tf.expand_dims(np.array([t]), 0))  # Update model hidden state
    return mdl.layers[1].states[0].numpy(), score


def makeIdxfromChar(str, char2idx, level):
    tokens = []
    if level=="CHAR":
        tokens=syl.get_chars(str)
    elif level=="SYL":
        tokens=syl.get_syllables(str)
    return [char2idx[t] for t in tokens]


# Beam state is (generated-text, fst-state, hidden-state, score)
def expandBeam(beam, fst, mdl, char2idx, level):
    #print("Expanding: "+beam[0])
    generated_text = beam[0]
    fst_state = beam[1]
    hidden_state = beam[2]
    score = beam[3]
    retBeams = []
    if fst_state == fst.states - 1:
        # If already printed the last token
        if generated_text[-8:] == "Couplet>":
            return [beam]
    lastLength=0
    for i,s in enumerate(fst.machine[fst_state]): # For each next state
        if len(fst.machine[fst_state])!=1:
            if i%100==0:
                printMsg="Expanding: "+str(i*100/len(fst.machine[fst_state]))
                print(printMsg)
        word, step = s
        textWord = fst.vocabulary[word][0]
        # update generated-text
        newText = generated_text + textWord
        # update fst-state
        newState = fst_state + step

        # update hidden-state and score
        # mdl.layers[1].reset_states(states=hidden_state)
        indexes = makeIdxfromChar(fst.vocabulary[word][0], char2idx, level)
        newH, newScore = getScore(mdl, hidden_state, indexes)

        # Continue adding tokens recursively to the beam if there is only one choice for the next state
        # Return this further expanded beam or beam with possible next expantions
        if len(fst.machine[newState]) == 1:
            nextWord, _ = fst.machine[newState][0]
            if fst.vocabulary[nextWord][0][-1] != ">" or fst.vocabulary[nextWord][0] == "<izafe>" or \
                    fst.vocabulary[nextWord][0] == "<mahlas>":
                # Add space if next token is not a <> invisible token
                newH, newnewScore = getScore(mdl, newH, [char2idx[" "]])
                newScore += newnewScore
                newText += " "
            newBeam = (newText, newState, newH, score + newScore)
            retBeams += expandBeam(newBeam, fst, mdl, char2idx, level)
        else:
            if textWord[-1] != '>' or textWord == "<izafe>" or textWord == "<mahlas>":
                # Add space if this word is not a <> invisible token
                newH, newScore = getScore(mdl, newH, [char2idx[" "]])
                newText += " "
            newBeam = (newText, newState, newH, score + newScore)
            retBeams.append(newBeam)

    return retBeams


def generate_text(model, char2idx, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

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

    tf.random.set_seed(1)
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

def beamArrayToStr(bArray):
    ret="\n"
    for b in bArray:
        ret+=b[0]+"\n"
    return ret


def selectBeams(beams, size):
    if len(beams)<size:
        return beams
    else:
        ret = sorted(beams, key=lambda b: b[3],reverse=True)[0:size]
        return ret


if __name__ == "__main__":
    LEVEL = "CHAR"
    BEAMSIZE = 5
    if LEVEL == "SYL":
        _, idx2char, char2idx = model.createSylLevelData("data/OTAP clean data/total-transcription")
    elif LEVEL == "CHAR":
        _, idx2char, char2idx = model.createCharLevelData("data/OTAP clean data/total-transcription", pad=True)
    else:
        print("Invalid LEVEL parameter")
        exit()

    checkpoint_path = "Code/Checkpoints/CharacterLevel/Transcription/ckpt50"
    langModel = model.buildGRUModelWithEmbedding(vocab_size=len(idx2char), embedding_dim=128, rnn_units=256,
                                                 batch_size=1,
                                                 statefulness=True)
    print(checkpoint_path)

    langModel.load_weights(checkpoint_path)
    langModel.build(tf.TensorShape([1, None]))
    langModel.summary()

    stateShape = langModel.layers[1].states[0].numpy().shape
    s_0 = np.zeros(shape=stateShape)
    langModel.layers[1].reset_states(states=s_0)

    #outp = "./data/OTAP clean data/wordList.txt"
    outp = "./data/tempWordList.txt"
    # FST.makeWordList("./data/OTAP clean data/total-transcription", outp)
    #vezn = FST.mefailunmefailun
    vezn=FST.failatun
    fst = FST.FST(vezn, outp)
    #fst.constrain(0, 'ki bil dil bil')
    #fst.constrain(1,'bil bil')
    print(str(fst))
    print("\n**********************\n")
    fst.reverse()
    # EMPLOY BEAM SEARCH
    beam = ("", 0, langModel.layers[1].states[0].numpy(), 0.0)
    #print(str(fst))
    beams = []
    beams.append(beam)
    cont = True
    while (cont):
        cont = False
        newBeams = []
        print(str(len(beams))+" beams")
        for b in beams:
            print("Expanding: '"+b[0]+"'")
            if b[1] < fst.states - 1:
                cont = True
            freshBeams = expandBeam(b, fst, langModel, char2idx, LEVEL)
            print("\n", end="")
            print("Received "+str(len(freshBeams))+" beams:"+beamArrayToStr(freshBeams))
            newBeams+=freshBeams
        beams = selectBeams(newBeams, BEAMSIZE)
    for i in range(len(beams)):
        print(str(i) + " - " + beams[i][0] + ",\t" + str(beams[i][1])+"\t"+str(beams[i][3]))
