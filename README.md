Get dataset and model here:
https://drive.google.com/drive/folders/0B14lJ2VVvtmJWEQ5RlFjQUY2Vzg

# sempar-codgen
Semantic Parsing and Code Generation project

Raphael Olivier, Pravalika Avvaru, Shirley Anugrah Hayati

##Organisation of the repository
* **data** : our datasets, Django and Hearthstone. Some files have been extracted with the instructions of their respective repositories, others (those we use here) were directly given by Pengcheng Yin, author of the paper [A Syntactic Neural Model for General-Purpose Code Generation](https://arxiv.org/abs/1704.01696).
### Checkpoint 1
Our baseline was inspired by the paper [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449). The files are organised as follow :
* **src** : our code
    * *extract_data.py* : a script used at some point (not for our final experiments) to extract the datasets from their raw forms.
    * *model.py* : specifies a class that implements our sequence-to-sequence with attention model. It was coded in DyNet, with the help of the documentation and the course code examples.
    * *baseline.py* : the script. To run it, run the command `python baseline.py --mode chosen_mode --iter n_iterations [other dynet options]` with `chosen_mode` being `django` or `hs` (Hearthstone). Other implementation parameters are hardcoded in the file and can easily be changed.
    * *accuracy.py* : a small script to compute accuracy and BLEU score. File paths are harcoded. The script uses nltk as a dependancy.
### Checkpoint 2
* **src** : our code
    * *grammar.py* : grammar model
    * *indexer.py* : extract rules, nodes, and vocabulary from data set using some of Pengcheng Yin's code into tsv file
    * *model1.py* : sepcifies a class that implements the neural model as implemented in the paper [A Syntactic Neural Model for General-Purpose Code Generation]. It was coded in Dynet, with the help of the documentataion and the course code examples
    * *sota.py* : the script. To run it, run the command `python sota.py --autobatching 1 --iter=10. Other implementation parameters are hardcoded in the file and can be easily changed.
    * *accuracy.py*: a small script to compute the accuracy and the BLEU score. File paths are hardcoded. The script uses nltk dependency.
    * *ast_grammar.py*: Contains the AST rules
    * *tree.py* : Python script to which interacts with the neural model and provide tree information during training and prediction
