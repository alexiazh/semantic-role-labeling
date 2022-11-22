## **Semantic Role Labeling using AllenNLP**

This script takes sample sentences which can be a single or list of sentences and uses AllenNLP's pre-trained model on Semantic Role Labeling to make predictions.

## **Description**

**Semantic Role Labeling**

Semantic Role Labeling (SRL) is the task of determining the latent predicate argument structure of a sentence and providing representations that can answer basic questions about sentence meaning, including who did what to whom, etc.

The model used for this script is found at https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz

The model card, and model usage can be found at https://demo.allennlp.org/semantic-role-labeling

## **Installlations**

Run ```pip install -r requirements.txt```.

This is the same as running ```pip install allennlp==2.1.0 allennlp-models==2.1.0```.

For other options, see https://github.com/allenai/allennlp#installation

## **Usage**

To run script with SRL model:

```python3 srl.py```

https://www.nltk.org/api/nltk.tag.html

## **English PropBank SRL**
AllenNLP uses English PropBank SRL Annotation. 
See more at https://hanlp.hankcs.com/docs/annotations/srl/propbank.html

| Role | Description                            |
|------|----------------------------------------|
| ARG0 | agent                                  |
| ARG1 | patient                                |
| ARG2 | instrument, benefactive, attribute     |
| ARG3 | starting point, benefactive, attribute |
| ARG4 | ending point                           |
| ARGM | modifier                               |
| COM  | Comitative                             |
| LOC  | Locative                               |
| DIR  | Directional                            |
| GOL  | Goal                                   |
| MNR  | Manner                                 |
| TMP  | Temporal                               |
| EXT  | Extent                                 |
| REC  | Reciprocals                            |
| PRD  | Secondary Predication                  |
| PRP  | Purpose                                |
| CAU  | Cause                                  |
| DIS  | Discourse                              |
| ADV  | Adverbials                             |
| ADJ  | Adjectival                             |
| MOD  | Modal                                  |
| NEG  | Negation                               |
| DSP  | Direct Speech                          |
| LVB  | Light Verb                             |
| CXN  | Construction                           |