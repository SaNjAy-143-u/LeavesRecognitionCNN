import argparse
import model
import loadDataset

def main_run(inp_path,outDir,img_size,momentum,lossfn,batchSize,numEpochs,noOfLayers,decay,learnRate,loadModel,plot):
    X_train,X_test,Y_train,Y_test,numClasses=loadDataset.loadDataset.getData(inp_path,img_size,loadModel,num_channel=1)
    model_obj=model.model(numClasses,noOfLayers,X_train[0].shape,momentum,lossfn,decay,learnRate)
    model_obj.forward(X_train,Y_train,X_test,Y_test,batchSize,numEpochs,outDir=outDir,loadModel=loadModel,Plot=plot)


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numEpochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--imgSize', type=tuple, default=(256,256), help='img_size')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--decay', type=float, default=0.06, help='Weight decay')
    parser.add_argument('--learnRate', type=float, default=0.01, help='Learning Rate')
    parser.add_argument('--batchSize', type=int, default=16, help='Training batch size')
    parser.add_argument('--noOfLayers', type=int, default=5, help='ConvLSTM hidden state size')
    parser.add_argument('--lossfn', type=str, default='categorical_crossentropy', help='loss function',choices=['categorical_crossentropy','sparse_categorical_crossentropy','poisson','mean_absolute_error','mean_squared_logarithmic_error'])
    parser.add_argument('--outDir', type=str, default='Data', help='Output directory')
    parser.add_argument('--inpDir', type=str, default=None, help='Directory containing  dataset')
    parser.add_argument('--loadModel', type=str, default=None, help='Directory containing model.hdf5')
    parser.add_argument('--plot', type=bool, default=False, help='Path to saved model')
    
    args = parser.parse_args()

    numEpochs = args.numEpochs
    imgSize=args.imgSize
    momentum = args.momentum
    decay=args.decay
    learnRate=args.learnRate
    batchSize = args.batchSize
    noOfLayers = args.noOfLayers
    lossfn = args.lossfn
    outDir = args.outDir
    inpDir = args.inpDir
    plot=args.plot
    loadModel = args.loadModel
    main_run(inpDir,outDir,imgSize,momentum,lossfn,batchSize,numEpochs,noOfLayers,decay,learnRate,loadModel,plot)

__main__()
