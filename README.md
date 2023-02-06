# Spell Correction using Language Model (LM)
This experiment uses the n-Gram language model trained on the news genre of the Brown corpus to find the correct spelling of misspelled words in the **Birkbeck** corpus. Where n={1, 2, 3, 5, 10} and k={1, 5, 10}, the average success at k, is calculated for each n.

**Keywords:** Spell correction, Language Model, Corpus, n-Gram, Probability, Natural Language Processing.

## The Data
The [APPLING1DAT.643](https://github.com/gloryodeyemi/COMP_8730_Assignment2/blob/main/Data/APPLING1DAT.643) file, out of the Birkbeck spelling error corpus by Roger Mitton was used for this experiment. They contain 198 entries of misspelled words in total and the correct equivalent of these words.

The Brown corpus contains 100554 words and 4623 sentences.

## Requirements
You can find the modules and libraries used in this project in the [requirement.txt](https://github.com/gloryodeyemi/COMP_8730_Assignment1/blob/main/requirements.txt) file. You can also run the code below.
```
pip install -r requirements.txt
```

## Structure
* **[Data](https://github.com/gloryodeyemi/COMP_8730_Assignment2/tree/main/Data):** contains the Birbeck corpus file used for this project.

* **[utils](https://github.com/gloryodeyemi/COMP_8730_Assignment2/tree/main/utils):** contains the essential functions for this project.

* **[models](https://github.com/gloryodeyemi/COMP_8730_Assignment2/tree/main/models): contains the trained models**

* **[Assignment_#2.ipynb](https://github.com/gloryodeyemi/COMP_8730_Assignment2/blob/main/Assignment_%232.ipynb)** and **[Assignment_#2.py](https://github.com/gloryodeyemi/COMP_8730_Assignment2/blob/main/Assignment_%232.py)** are python notebook and script that uses the functions in the utils folder to generate the results.

## Contact
Glory Odeyemi is currently undergoing her Master's program in Computer Science, Artificial Intelligence specialization at the [University of Windsor](https://www.uwindsor.ca/), Windsor, ON, Canada. You can connect with her on [LinkedIn](https://www.linkedin.com/in/glory-odeyemi-a3a680169/).

## References
1. [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus)
2. [Birkbeck spelling error corpus](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/0643)
3. [PyTrec-Eval-Terrier](https://pypi.org/project/pytrec-eval-terrier/)
