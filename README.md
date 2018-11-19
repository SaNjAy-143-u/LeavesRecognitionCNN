# LeavesRecognitionCNN

This is an implementation of paper [A Convolutional Neural Network for Leaves Recognition Using Data Augmentation]
(https://ieeexplore.ieee.org/document/7363364).

#### Prerequisites
* Python 3.6
* keras 2.2.4
#### Running

```
python main-run.py --numEpochs 100 \
--imgSize (256,256) \
--momentum 0.9 \
--decay 0.06 \
--learnRate 0.01 \
--batchSize 80 \
--noOfLayers 5\
--lossfn 'categorical_crossentropy'\
--outDir 'outData'\
--inpDir './data'\
```
