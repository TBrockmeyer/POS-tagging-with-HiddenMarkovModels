# Part-of-Speech tagging with Hidden Markov Models

This project is part of an NLP course at Universität Würzburg.
The task is to analyze a given sentence and propose part-of-speech tags, i.e. categories of words.
E.g., if the underlying model has been trained with a sample of German words and tags as in the file "training_pos.txt" delivered with this project, for the sentence

"Wir rennen oft zum Bus ."

it shall propose a sequence of ideally correct tags:

PPER VVFIN ADV APPRART NN $.

-20.2232178033

(The floating point number is the log likelihood of the tag sequence proposed)

A widely used tag set for the German language can be obtained here: www.ims.uni-stuttgart.de/forschung/ressourcen/lexika/TagSets/stts-table.html

## Getting Started

You need a Python 3 development environment installed on your computer, as well as the numpy, math and argparse packages for Python.
Download at least hmm.py and, if you don't have your own training set of word-tag pairs, training_pos.txt into a project folder.

### Running the code and analyzing sentences

For training (i.e., creating) a model from a given text file of word-tag pairs, open a command window in the same folder as the hmm.py file and use the following command

python hmm.py train text_file_name.txt model_name.npz

(For trying, use the file "training_pos.txt" instead of text_file_name.txt, but feel free to use other training sets from any language.
You can choose another name for model_name.npz, but always use the ending ".npz"!)
For classfying a sentence, use the command

python hmm.py classify existing_model_name.npz "sentence to be classified, in quotation marks"

(The .npz file must be an existing model [created e.g. with command above], the sentence should be in the same language as the training word-tag sample used for the .npz model)
