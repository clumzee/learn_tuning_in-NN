# learn_tuning_in-NN
# **LEARN HYPER PARAMETER TUNING IN NEURAL NETWORKS WITHOUT WRITING A SINGLE LINE OF PYTHON CODE**

This Github repository was made as a part of a challenge given by Ineuron team to me.


All you need to do is to download the requirements.txt file and run the command "pip install -r requiremtns.txt"
Now run the train.py module.

You can run train.py module without any argument or you can change in total 7 hyperparameters.

Theses Hyper parameters are 
  1. Number of layers in the Model (default:3)
  2. Number of Neurons in each layer (default:128)
  3. activation function of every layer except the output layer (Softmax) (default:relu)
  4. optimizer function (default:adam)
  5. number of epochs (default:5)
  6. Loss function (default:sparse_categorical_crossentropy)
  7. batch_norm (i.e to include Batch normalisation layer or not) (default:0, means false)

Some points to remember are that
  1. This module works on *MNIST* dataset only as per given by the challenge guideline. (Will try to upgrade this feature to support more datasets in fututre)
  2. The last parameter batch_norm will add batch normalisation layer after every 3 layers
  3. Although you can put any number in the first parameter (i.e number of layers) but be sensible with it as it can possibly crash your P.C. Suggested range would be 2-5
  4. You have to put the parameters in the above given order only, otherwise it won't work one valid CMD command will be "python train.py 4 128 relu adam 5 sparse_categorical_crossentropy 0"
  5. Create 2-3 models like this and finally run the command "mlflow ui" and now you can see all the models built by you and compare different results created by different parameters.
 
