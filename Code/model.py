import syllabliser as syl
import numpy as np
import tensorflow as tf
import os

def prepareForTraining(x):
    return x[:-1],x[1:]

def get_chars(str):
    chrs=list()
    c = 0
    while c < len(str):
        if str[c] == "<":
            c2 = c
            while str[c2] != ">":
                c2 += 1
            chrs.append(str[c:c2 + 1])
            c = c2
        else:
            chrs.append(str[c])
        c += 1
    return chrs

def createCharLevelData(fName, pad=False):
    f=open(fName)
    lines=f.readlines()
    f.close()
    charDataset=list()
    for l in lines:
        l = l.replace("s̲", "S")
        couplet=list()
        spl=l.split()
        for s in range(0, len(spl)):
            if spl[s] == "<beginCouplet>" or spl[s]=="<endLine>" or spl[s]=="<endCouplet>":
                couplet.append(spl[s])
            else:
                chars=get_chars(spl[s])
                for c in chars:
                    couplet.append(c)
                if spl[s + 1][0] != "<" or spl[s + 1][0:8] == "<mahlas>":  # It is sure that spl[s+1] exists because if not, then spl[s]=="<endCouplet>" and control doesn't enter this if
                    couplet.append(" ")
        charDataset.append(couplet)
    flatText = [charac for coupl in charDataset for charac in coupl]
    #for s in charDataset[0:203]:
    #    print(s)
    if pad:
        vocab = ["<PAD>"] + sorted(set(flatText))
    else:
        vocab=sorted(set(flatText))
    # print(len(vocab))
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = np.array(vocab)
    idxDataset = list()
    maxLength = np.amax(np.array([len(l) for l in charDataset]))
    for coup in charDataset:
        cplt = [char2idx[c] for c in coup]
        pad_len = maxLength - len(cplt)
        cplt = np.array(cplt)
        if pad:
            cplt = np.pad(cplt, (0, pad_len), constant_values=(0, 0))
        idxDataset.append(cplt)
    return np.array(idxDataset), idx2char, char2idx



def createSylLevelData(fName):
    f = open(fName)
    lines = f.readlines()
    f.close()
    sylDataset = list()
    for l in lines:
        couplet = list()
        spl = l.split()
        for s in range(0, len(spl)):
            if spl[s] == "<beginCouplet>" or spl[s]=="<endLine>" or spl[s]=="<endCouplet>":
                couplet.append(spl[s])
            else:
                syls=syl.get_syllables(spl[s])
                for sy in syls:
                    couplet.append(sy)
                if spl[s + 1][0] != "<" or spl[s + 1][0:8] == "<mahlas>":  # It is sure that spl[s+1] exists because if not, then spl[s]=="<endCouplet>" and control doesn't enter this if
                    couplet.append(" ")
        sylDataset.append(couplet)
    flatText = [syllable for coupl in sylDataset for syllable in coupl]
    #for sss in sylDataset[0:203]:
    #    print(sss)
    vocab = ["<PAD>"]+sorted(set(flatText))
    #print(len(vocab))
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = np.array(vocab)
    idxDataset = list()
    maxLength = np.amax(np.array([len(l) for l in sylDataset]))
    for coup in sylDataset:
        cplt = [char2idx[c] for c in coup]
        pad_len = maxLength-len(cplt)
        cplt = np.array(cplt)
        cplt = np.pad(cplt, (0, pad_len), constant_values=(0,0))
        idxDataset.append(cplt)
    return np.array(idxDataset), idx2char, char2idx


def createOTAPDataFromIndividualTexts(latinCharSet=False):
    names=["mihri","necati","revani-all"]
    data=list()
    extent=0
    for i in names:
        f=open("data/OTAP clean data/"+i)
        linez=f.readlines()
        f.close()
        dataUnit=list()
        for l in linez:
            l = l.strip().lower()
            cont=True
            if extent == 0:
                if l == "4":
                    extent = 4
                    cont=False
                elif l == "5":
                    extent = 5
                    cont=False
                else:
                    extent = 2
            if cont:
                l = l[3:]
                dataUnit.append(l)
                extent = extent - 1
                if extent==0:
                    data.append(dataUnit)
                    dataUnit=[]
    f=open("data/OTAP clean data/total", "w")
    for s in data:
        f.write("<beginCouplet> ")
        for l in s:
            #l=l.replace(" ", " <space> ")
            if latinCharSet:
                l=l.replace("ā", "â")
                l=l.replace("ī", "î")
                l=l.replace("ō", "ô")
                l=l.replace("ū", "û")
                l=l.replace("ż", "z")
                l=l.replace("ẓ", "z")
                l=l.replace("ẕ", "z")
                l=l.replace("s̲", "s")
                l=l.replace("ṣ", "s")
                l=l.replace("ḍ", "d")
                l=l.replace("ḥ", "h")
                l=l.replace("ẖ", "h")
                l=l.replace("ḫ", "h")
                l=l.replace("ḳ", "k")
                l=l.replace("ṭ", "t")
                l=l.replace("ṯ", "t")
                l=l.replace("ñ", "n")
            f.write(l)
            f.write(" <endLine> ")
        f.write("<endCouplet>\n")
    f.close()

def buildGRUModelWithEmbedding(vocab_size, embedding_dim, rnn_units, batch_size, masking=True, statefulness=False):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,mask_zero=masking, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,return_sequences=True, stateful=statefulness, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
        ])
    return model

if __name__=="__main__":
    createOTAPDataFromIndividualTexts(False)

    LEVEL="CHAR"
    if LEVEL=="SYL":
        data,idx2char,char2idx=createSylLevelData("data/OTAP clean data/total")
    elif LEVEL=="CHAR":
        data,idx2char,char2idx=createCharLevelData("data/OTAP clean data/total",pad=True)
    else:
        print("Invalid LEVEL parameter")
        exit()

    inputs=tf.data.Dataset.from_tensor_slices(data)
    dataset=inputs.map(prepareForTraining)
    cum=0
    for r in data:
        cum+=len(r)
    print(np.array(list(dataset.as_numpy_iterator())).shape)


    for input_example, target_example in dataset.take(1):
        print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
        print('Target data:', repr(''.join(idx2char[target_example.numpy()])))
    #for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    #    print("Step {:4d}".format(i))
    #    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    #    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


    BATCH_SIZE=100

    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=False)

    model=buildGRUModelWithEmbedding(vocab_size=len(idx2char), embedding_dim=128, rnn_units=256, batch_size=BATCH_SIZE)
    model.summary()
    for input_example_batch, target_example_batch in dataset.take(1):
        print(input_example_batch.shape)
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    print(sampled_indices)

    print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
    print()
    print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))


    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())

    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = 'checkpoints/syl-level/'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True, monitor="loss", save_best_only=True, mode="min")

    EPOCHS = 50
    for i in char2idx.keys():
        print(i)
    #hist = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])"""