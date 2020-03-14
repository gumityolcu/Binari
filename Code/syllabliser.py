"""

Türkçede kelime içinde iki ünlü arasındaki ünsüz,
kendinden sonraki ünlüyle hece kurar: a-ra-ba, bi-çi-mi-ne, in-sa-nın, ka-ra-ca vb.

Kelime içinde yan yana gelen iki ünsüzden ilki kendinden önceki ünlüyle,
ikincisi kendinden sonraki ünlüyle hece kurar: al-dı, bir-lik, sev-mek vb.

Kelime içinde yan yana gelen üç ünsüz harften ilk ikisi kendinden önceki ünlüyle,
üçüncüsü kendinden sonraki ünlüyle hece kurar: alt-lık, Türk-çe, kork-mak vb.

"""


def isVowel(c):
    return (c == "a") or (c == "ā") or (c == "e") or (c == "ı") or (c == "i") or (c == "ī") or (c == "o") or (
            c == "ö") or (c == "ū") or (c == "u") or (c == "ü")


def get_syllables(word):
    if len(word) < 3:
        return [word]
    inSyllable = [-1] * len(word)
    vowelCount = 0
    vowelPoss = list()
    syllables = list()
    for c in range(0, len(word)):
        if isVowel(word[c]):
            vowelCount = vowelCount + 1
            inSyllable[c] = vowelCount
            vowelPoss.append(c)
            syllables.append(word[c])
    inSyllable[0] = 1
    if not isVowel(word[0]):
        syllables[0] = word[0] + syllables[0]

    inSyllable[-1] = vowelCount
    if not isVowel(word[-1]):
        syllables[-1] = syllables[-1] + word[-1]
    for c in range(1, len(word) - 1):
        if not isVowel(word[c]):
            if isVowel(word[c + 1]):
                inSyllable[c] = inSyllable[c + 1]
                syllables[inSyllable[c] - 1] = word[c] + syllables[inSyllable[c] - 1]
            else:
                inSyllable[c] = inSyllable[c - 1]
                syllables[inSyllable[c] - 1] = syllables[inSyllable[c] - 1] + word[c]
    return syllables


def isLong(c):
    return (c == "ā") or (c == "ī") or (c == "ū")


def elongateMetre(met):
    longMetre = []
    for a in met:
        if a == "-.":
            longMetre.append("-")
            longMetre.append(".")
        else:
            longMetre.append(a)
    return longMetre


def get_aruz(word):
    word = word.replace('·', '')
    word = word.replace('-', '')
    word = word.replace('\'', '')
    sylls = get_syllables(word)
    aruz = list()
    for i in sylls:
        if isVowel(i[-1]):
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

