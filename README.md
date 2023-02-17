# Checklist
- [  ] NLP theory like encoders, decoders, transformers, LSTM, BERT etc.
- [  ] Pytorch and Pytorch Lighting
- [  ] In depth Literature Review
- [  ] Internal Working of CNN (Math)
- [  ] Mathematics behind various optimizers
- [  ] Various types of neural networks, their workings (math)
    - [  ] RNN and ANN
    - [  ] Feedforward
    - [  ] MLP
    - [  ] ResNet
    - [  ] CNN
    - [  ] LSTM

# Introduction
# Motivation
# Generic Architecture
Traditional architcture that do not perform well in the domain include unimodal architecture that process only either image or text in the image and give bad results. However, recent studies have shown the success of multimodal architectures that combine the meaning of text with the content in the image.

As a result, even though there are slight variations, the generic architecture of the majority of multimodal models remain the same. The general architecture has two types of flows i.e Linguistic processing flow (LPF) and visual processing flow (VPF). There is always an integrating phase which is called fusion and pre-training phase (FPT) which defined strategies for merging LPF and VPF. The merging phase is usually the part of the architecture that provides a decision.

### Linguistic Processing Flow (LPF)
* Four steps include preprocessing, feature engineering, dimensionality reduction and classification.
* Preprocessing incldues stop words removal, capitalization, tokenization, abbreviation handling, spelling correction, noise removal, stemming and lemmatization.
* Feature engineering includes word embeddings like word2vec, glove, n-gram, bag of words, tf-idf etc.
* Dimensionality reduction includes PCA, LDA, ICA etc.
* For 1G LPF model selection includes choice likfe SVM, knn, naive bayes etc.
* For 2G LPF model selection includes neural network based models like BERT, GPT, LSTM
* Attention layers including transformer, encoders and decoders are also used to normalize the calculated matching score between query vector and each context vector among all vectors using softmax.

### Visual Processing Flow (VPF)
* Feature extraction methods like LBP, SIFT, HOG,SURF, BRIEF etc have been used.
* Various pretrained models like AlexNet, GoogleNet, VGG etc having the capabillity of transfer learning has changed the course of VPF.
* Furthermore, the recent advances in DenseNet reformulated the connections between network layers that further boost the learning and representational properties of deep networks.

### Fusion and Pre-training
* Three categories includin early fusion, late fusion and hybrid fusion.
* Early fusion merges the features instantly after they are extracted.
* Late fusion integrates the decisions after each modality has taken its decision.
* Hybrid fusion fuse outpus from individual unimodal predictors.