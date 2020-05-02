def OTAPAnalyser(fName, writeToFile):
    path = "./data/" + fName
    f = open(path)
    lines = f.readlines()
    f.close()
    vocab = dict()
    extent = 0
    for l in lines:
        cont = True
        l = l.strip().lower()
        if extent == 0:
            if l == "4":
                extent = 4
                cont = False
            elif l == "5":
                extent = 5
                cont = False
            else:
                extent = 2
        if cont:
            extent = extent - 1
            l = l[3:]
            spl = l.split(" ")
            for s in spl:
                if s in vocab:
                    vocab[s] += 1
                else:
                    vocab[s] = 1
    if writeToFile:
        path="./data/"+fName+"-words"
        output = open(path,"w")
        k=sorted(vocab.keys())
        for s in k:
            output.write(s+"\t"+str(vocab[s])+"\n")
        output.close()
    return vocab

def safahatAnalyser(writeToFile=True):
    path = "./data/safahat-edited"
    f = open(path)
    lines = f.readlines()
    f.close()
    vocab = dict()
    extent = 0
    for l in lines:
        cont = True
        l = l.strip().lower()
        l=l.replace(",","")
        l=l.replace(":","")
        l=l.replace(".","")
        l=l.replace("!","")
        l=l.replace("?","")
        l=l.replace("(","")
        l=l.replace(")","")
        l=l.replace(";","")
        l=l.replace("\"","")
        l=l.replace("-","")
        spl = l.split(" ")
        for s in spl:
            if not s=="":
                if s in vocab:
                    vocab[s] += 1
                else:
                    vocab[s] = 1

    if writeToFile:
        path="./data/safahat-words"
        output = open(path,"w")
        k=sorted(vocab.keys())
        for s in k:
            output.write(s+"\t"+str(vocab[s])+"\n")
        output.close()
    return vocab

def standardise(fName):
    path = "./data/" + fName
    f = open(path)
    lines = f.readlines()
    f.close()
    for l in lines:
        l2=l.lower()
        l2=l2.replace("ā","â")
        l2=l2.replace("ā","â")
        l2=l2.replace("ā","â")
        l2=l2.replace("ā","â")

a="asd"

b=["asd"]
for c in a:
    b.append(c)
print(b)