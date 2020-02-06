# Pos-Tagger
Α Pos Tagger trained on UD treebank with fine-tuning a BERT model

### network model structure

![model](model.png)

# Environment
- Keras==2.2.5
- nltk==3.2.5  
- pyconll
- pydot
- graphiz
- bert-tensorflow

# Dataset
Download dataset run this script or download this [link]()
```py
def download_files():
    print('Downloading English treebank...')
    urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-dev.conllu', 'en_partut-ud-dev.conllu')
    urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-test.conllu', 'en_partut-ud-test.conllu')
    urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-train.conllu', 'en_partut-ud-train.conllu')
    print('Treebank downloaded.')
download_files()

```
# Train pos tagger model
Run script ``` train.py ```

# Model inference 
Download pretrain model [here]() and run below this script ```inference.py```

