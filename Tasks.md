# Tasks
- [ ] Improve the accuracy of model
    - [x] Change model layers / architecture
    - [x] Resnet150 with transfer learning
    - [x] Change dimension params
    - [x] Change Learning Rate
    - [ ] Use an entirely different encoder-decoder architecture
    - [ ] Use hyperparameter optimization
- [ ] OCR module and inference pipeline
    - [x] Research on various techniques in OCR
    - [ ] Use trial version of Google OCR
- [ ] Create a fast api
- [ ] Deploy on free cloud services
- [ ] Frontend demo

#### Methods to improve the accuracy of the model
* Fine-Tune individual models
* Use ensemble learning
* Hyperparameter tuning
* Change model architecture including change encoder-decoder.

# Suggestions
* Fine tune individual models
* Perform hyperparameter optimization
    * Learning Rate
    * Batch Size
    * Dropout Rate
    * Number of neurons in a dense layer
* See how each individual model is performing.
* Training the models individually and then running the ensemble also provides high level of accuracy.

# Small Tasks that are left
* Modify tasks to get AUROC score
* Debug training and validation loss = 0