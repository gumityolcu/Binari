"""

Türkçede kelime içinde iki ünlü arasındaki ünsüz,
kendinden sonraki ünlüyle hece kurar: a-ra-ba, bi-çi-mi-ne, in-sa-nın, ka-ra-ca vb.

Kelime içinde yan yana gelen iki ünsüzden ilki kendinden önceki ünlüyle,
ikincisi kendinden sonraki ünlüyle hece kurar: al-dı, bir-lik, sev-mek vb.

Kelime içinde yan yana gelen üç ünsüz harften ilk ikisi kendinden önceki ünlüyle,
üçüncüsü kendinden sonraki ünlüyle hece kurar: alt-lık, Türk-çe, kork-mak vb.

"""
vowels = ["a", "ā", "â", "e", "ı", "i", "ī", "î", "o", "ö", "ô", "u", "ū", "û", "ü"]
longVowels = ["ā", "â", "ī", "î", "ô", "ū", "û"]

def isVowel(c):
    return c in vowels

def isLong(c):
    return c in longVowels

def get_syllables(word):
    if len(word) < 3:
        return [word]
    # We will assign each character to a syllable in the array inSyllable
    inSyllable = [-1] * len(word)
    vowelCount = 0
    vowelPoss = list()
    syllables = list()
    # Count the number of vowels to find the number of syllables
    for c in range(0, len(word)):
        if isVowel(word[c]):
            vowelCount = vowelCount + 1
            inSyllable[c] = vowelCount
            vowelPoss.append(c)
            # Add the vowels of the syllables
            syllables.append(word[c])

    # First character belongs to the first syllable
    inSyllable[0] = 1
    if not isVowel(word[0]):
        # Add the character to the first syllable if not already added
        syllables[0] = word[0] + syllables[0]

    # Last character belongs to the last syllable
    inSyllable[-1] = vowelCount
    if not isVowel(word[-1]):
        # Add the character to the last syllable if not already added
        syllables[-1] = syllables[-1] + word[-1]

    # Assign syllables and edit them according to the logical rule
    for c in range(1, len(word) - 1):
        if not isVowel(word[c]):
            if isVowel(word[c + 1]):
                inSyllable[c] = inSyllable[c + 1]
                syllables[inSyllable[c] - 1] = word[c] + syllables[inSyllable[c] - 1]
            else:
                inSyllable[c] = inSyllable[c - 1]
                syllables[inSyllable[c] - 1] = syllables[inSyllable[c] - 1] + word[c]

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

