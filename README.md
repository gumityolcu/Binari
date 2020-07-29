# Binari
This is a repository for Senior Project in Boğaziçi University Computer Engineering Department.

The project's aim is to write a computer program that generates poems called "Gazel"s in Ottoman Turkish.

Digitised ghazal poems from the [Ottoman Text Archive Project](http://courses.washington.edu/otap/) is used. The results and the techniques can be found in Presentation/Binari_Report.pdf
/
Related Wikipedia pages:

[Diwan(poetry)](https://www.wikiwand.com/en/Diwan_(poetry))

[Ghazal](https://www.wikiwand.com/en/Ghazal)

[Ottoman poetry](https://www.wikiwand.com/en/Ottoman_poetry)

# Repository Contents

The main directory contains the papers which describe the main technique used in this study.

The *data* directory contains the text files used for training. Both the original versions, and the refactored/standardised versions with the list of changes made in the texts.

The *Presentation* directory contains the report and the poster for the project.

The *Code* directory contains:
* syllabliser.py which contains functions that find syllables or aruz representations of words
* FST.py which contains an FST class that accepts a list of words and generates and FST with them in the spirit of the papers. YOu can then constrain the FST with rhyme words or reverse it etc.
* model.py which is used for model training
* Binari.py and Binariv2.py which are used to generate poems using the saved neural network parameters and FSTs. Binari.py uses word level FST whereas Binariv2.py doesn't use an FST, it uses a syllable level RNN and chooses syllables that comply with the rhythmic metre.
* Checkpoints of training, i.e. model parameters
* Data Analysis folder used for scraping some poems online and computing some statistics from the data. It is not essential for the product.
