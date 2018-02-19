Kernelised Dynamic Mixing
=========================

A collection of methods for dynamically switching between kernel and model-based planning. Included in the 'toy' folder:
  - a script to create a toy data set of a certain size
  - training, test and validation data sets with corresponding fencepost sets indicating where each sequence starts and ends
  
To run KDM, first open the run.py file. In the get_parser() function, make sure that the default settings for the arguments '--dataset' and '--out_dir' are set to the location of the dataset and the directory to output to respectively.

Once the dataset and output directories have been set, run KDM by executing

*python run.py*

Alternatively, you can set these parameters by executing

*python run.py --dataset d --out_dir o* from the command line.

This will produce a list of policies for the data set.

Please reach out to sonali.parbhoo@unibas.ch for any bug reports or concerns.






