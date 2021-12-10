# EnronEmailGasOil

(I) Objectives of the Project
========================================

Enron was a major electricity, nautral gas, communication company which also had significant financial businesses for energy derivatives. It went bankrupt after its scandal of hiding masssive trading losses. This dataset containing around 0.5M emails were released by the Ferderal Energy Regulatory Commission during the investigation of its bankrupt and it can be downloaded from https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz

The target of this project is to build a model to identify which emails from the Enron dataset are related to gas/oil business. 

(II) Strategy to tackle the problem 
========================================

The emails downloaded can be parsed by the standard Python Parser for their subject and content. 

For the heuristic purpose of training the classfication models, the ground true labels (whether they are related to the oil/gas business) are given by whether the emails contain a term in the oil and gas industry glossary ^ (https://www.dwasolutions.com/images/DWA_Oil_Gas_Glossary.pdf ). It has been compared with a small set (200 emails) of texts labelled by hand and the portion of emails with relevant labels are similar (\~36.8 %). 

For an aggregate understanding of the emails with 'relevant' and 'irrelevant' labels, the wordclouds for them are plotted:

<img src="./wordcloud.png" width="600" >

In the 'relevant' emails, we can see there are more city names like 'new york' and 'London' and business related terms like 'scheduled outage' and 'natural gas'.

To begin with, a BiLSTM model with embedding in Keras is used for the email classfication. The embedding in Keras does not give any semantic meaning to the embedded word vectors. This is done by model 2a.

Then GloVe embedding is used with the BiLSTM classification in the model 2b's. The embedded vectors by GloVe has semantic meaning. But GloVe is pretrained with some general purpose text and our emails might be more specific to the custom field of gas/oil. So the weights in the embedding in model 2bii and indeed it gives a much higher accuracy than 2bi whose embedding weights are fixed.

The problems with GloVe are that it cannot deal with the polysemy of words and the word in our dataset may not be covered in its pretrained set of words might. BERT uses WordPiece tokenization scheme which breaks words into subwords and is better deal with these problems. Also its multi-head attention structure makes it the state-of-art NLP model. Indeed the test accuracy can boosted with BERT in our case. Model 2ci uses the original BERT model with one-layer fully connected layer for classification. Since a more complex classification head might be able to deal more more complex task, model 2cii uses a two-layer fully connected classfication head. In both model 2ci and 2cii, the weights in BERT are allowed to be updated. It is useful since BERT is pretrained with general purpose text. Indeed in model 2ciii where the weights in BERT were frozen, the performance is a lot worse.


Remarks: <br>
^ A more proper way for this is to do a weakly supervised learning, when time and resources allow. We can first label certain number of texts (say 1000-2000 of them) by hand and define some labelling functions to further label other texts with some ‘noisy labels’, which can be implemented by the Snorkel library. It has been shown that the accuracy of the classification model can be boosted when the data with teh amount of the noisy labelled data is up to 10x to 50x the hand labelled data.


(III) Test Results (Accuracy)
========================================

2a_BiLSTM.ipynb-> test: 93.37% \*

2bi_GloVe_Feature_Extraction(BiLSTM).ipynb-> test: 87.1% <br>
2bii_GloVe_fine-tuning(BiLSTM).ipynb-> test: 94.08% \* <br>

2ci_Bert_pretrained_head.ipynb-> test: 94.20% \* <br>
2cii_Bert_custom_head.ipynb-> test: 94.73% \* <br>
2ciii_Bert_custom_head_fine_tune_only.ipynb-> test: 74.26% 

\* models picked to be included in the Ensemble model in 2d below

2d_EnsembleMethod.ipynb-> test: 94.82% (Best with a subset with models 2bii, 2ci, 2cii)


(IV) Structure of the Directories
========================================

```
EnronEmailGasOil
│
├── 0_Convert_glossary_to_npz.ipynb
├── glossary.txt # the oil and gas glossary duplicated from the .pdf above
├── glossary.npz # a collection of oil and gas terms extracted from the .txt above
│
├── 1_Data_Exploration_Cleaning.ipynb # Organizing, exploring and cleaning the emails downloaded, also giving the ground true labels to the emails (saved in the .csv below) and putting the texts in the right formats for inputs to the models in 2's below (put in .npz below)
├── sampled_data.csv # store 100000 sampled emails and the ground true lable given to them (for train/val/test of the models)
├── ^ data_prep.npz # data from sampled_data.csv organized for feeding the BiLSTM models
├── ^ data_prep_bert.npz # data from sampled_data.csv organized for feeding the BERT models
│
├── 2a_BiLSTM.ipynb # classification by BiLSTM model using embedding in Keras 
│
├── 2bi_GloVe_Feature_Extraction(BiLSTM).ipynb # with GloVe pretrained embeddings
├── 2bii_GloVe_fine-tuning(BiLSTM).ipynb # allow the GloVe pretrained embeddings to be fine tuned here
│
├── 2ci_Bert_pretrained_head.ipynb # with pretrained BERT using its orginal one-layer fully connected layer for the classfication
├── 2cii_Bert_custom_head.ipynb # same as 2ci except the classification is by two fully connected layers, which is found to perform better
├── 2ciii_Bert_custom_head_fine_tune_only.ipynb # same as 2cii except the parameters in BERT are not allowed to be fine-tuned here
│
├── 2d_EnsembleMethod.ipynb # Ensemble methods using a collections of the models from 2a to 2c above
│
├── ^ maildir ** # a directory storing the emails for the Enron dataset
├── ^ GloVe_dict ** # a directory storing the pretrained GloVe embedding
└── ^ model_param # a directory storing the weights of the trained models in .h5 format
```

^ files/ directories that were not put on this depository since they are too big and their content can either be downloaded from the link indicated below or generated in the relevant .ipynb. The filenames shown here are for completion.

** maildir: the Enron dataset can be downloaded from https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz <br>
** GloVe_dict: the pretrained GloVe embedding which can be downloaded from https://nlp.stanford.edu/projects/glove/
 

