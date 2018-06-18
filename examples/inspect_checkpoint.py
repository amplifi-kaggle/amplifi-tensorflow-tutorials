#!/usr/bin/env python2

from tensorflow.python.tools import inspect_checkpoint as chkp

'''

We can quickly inspect variables in a checkpoint with the inspect_checkpoint library.

'''

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file("./model.ckpt", tensor_name='', all_tensors=True)

# print only tensor v1 in checkpoint file
chkp.print_tensors_in_checkpoint_file("./model.ckpt", tensor_name='v1', all_tensors=False)

# print only tnesor v2 in checkpoint file
chkp.print_tensors_in_checkpoint_file("./model.ckpt", tensor_name="v2", all_tensors=False)
