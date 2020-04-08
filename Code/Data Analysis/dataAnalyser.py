def OTAPAnalyser():
    f = open("./data/mihri")
    lines = f.readlines()
    f.close()
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

OTAPAnalyser()