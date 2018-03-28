# Deep-Learning

## Tensorflow vs Pytorch

When loading data using the data.py module, it is important to know whether 
Tensorflow or Pytorch is used. Tensorflow uses NHWC for it's batch tensor 
layout (i.e. batch.size() = [batch_size, height, width, nb_channels]) an 
Pytorch used NCHW. Specify the layout using the 'nwhc' option while calling 
the load function. 