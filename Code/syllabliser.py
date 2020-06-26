"""

Türkçede kelime içinde iki ünlü arasındaki ünsüz,
kendinden sonraki ünlüyle hece kurar: a-ra-ba, bi-çi-mi-ne, in-sa-nın, ka-ra-ca vb.

Kelime içinde yan yana gelen iki ünsüzden ilki kendinden önceki ünlüyle,
ikincisi kendinden sonraki ünlüyle hece kurar: al-dı, bir-lik, sev-mek vb.

Kelime içinde yan yana gelen üç ünsüz harften ilk ikisi kendinden önceki ünlüyle,
üçüncüsü kendinden sonraki ünlüyle hece kurar: alt-lık, Türk-çe, kork-mak vb.

"""
vowels = ["a", "ā", "ǎ", "â", "e", "ı", "i", "ī", "î", "o", "ö", "ô", "ō", "u", "ū", "û", "ü"]
longVowels = ["ā", "â", "ī", "î", "ô", "ū", "û"]


def isVowel(c):
    return c in vowels


def isLong(c):
    return c in longVowels


def get_syllables(word):
    if len(word) < 3:
        return [word]
    if word[0:8]=="<mahlas>":
        ret=["<mahlas>"]
        if word[8:]!='':
            ret+=get_syllables(word[8:])
        return ret
    if word=="<izafe>" or word=="<beginCouplet>" or word=="<endLine>" or word=="<endCouplet>":
        return [word]
    # We will assign each character to a syllable in the array inSyllable
    inSyllable = [-1] * len(word)
    vowelCount = 0
    vowelPoss = list()
    # Count the number of vowels to find the number of syllables
    for c in range(0, len(word)):
        if isVowel(word[c]):
            vowelCount = vowelCount + 1
            inSyllable[c] = vowelCount
            vowelPoss.append(c)

    # First character belongs to the first syllable
    inSyllable[0] = 1

    # Last character belongs to the last syllable
    inSyllable[-1] = vowelCount

    # Assign syllables and edit them according to the logical rule
    for c in range(1, len(word) - 1):
        if not isVowel(word[c]):
            if word[c]=="'":
                # ' symbol is always in the second position after the letter n in our data, thus the index c-1 will never go below 0
                inSyllable[c] = inSyllable[c-1]
            elif isVowel(word[c + 1]):
                inSyllable[c] = inSyllable[c + 1]
            else:
                inSyllable[c] = inSyllable[c - 1]

    # Construct syllables from inSyllable array
    syllables = [""]*vowelCount
    for i in range(0,len(inSyllable)):
        syllables[inSyllable[i]-1]=syllables[inSyllable[i]-1]+word[i]
    return syllables


def elongateMetre(met):
    longMetre = []
    for a in met:
        if a == "-.":
            longMetre.append("-")
            longMetre.append(".")
        else:
            longMetre.append(a)
    return longMetre


def get_chars(str):
    chrs = list()
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

def get_aruz(word):
    word = word.replace('·', '')
    word = word.replace('-', '')
    word = word.replace('\'', '')
    word = word.replace(' ', '')
    sylls = get_syllables(word)
    aruz = list()
    for i in sylls:
        if i == "<mahlas>":
            aruz+=[".","-","-"] # . - - for Binârî
        elif i=="<izafe>":
            aruz.append(".") # This is a placeholder, the <izafe> tokens is not compared against the actual metre in the FST implementation
        elif isVowel(i[-1]):
            if not isLong(i[-1]):
                aruz.append(".")
            else:
                aruz.append("-")
        else:
            if (not isVowel(i[-2])) or (isLong(i[-2]) and i[-1] != "n"):
                aruz.append("-.")
            else:
                aruz.append("-")
    return aruz

if __name__=="__main__":
    print(get_syllables("bîrışk"))
    print(get_syllables("abicim"))
    print(get_syllables("nasıl"))
    print(get_syllables("âbidelîk"))
    print(get_syllables("dünyadüzmü"))
    print(get_syllables("altıbuçulmilyon"))
    print(get_syllables("yüzdeiki"))
