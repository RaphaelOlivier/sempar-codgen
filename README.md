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
TODO
