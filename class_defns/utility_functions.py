import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
# import torch.nn as nn
# import class_defns.classical_head_classes as chc
# import class_defns.quantum_circuit_classes as qcc
# import class_defns.RL_classes as rlc
import numpy as np
# import copy
import matplotlib.pyplot as plt


def loading_dot_pt_files(filename):
    return(torch.load(filename,weights_only=False))

def plotting_testing_plot(each_iteration_tensor:torch.Tensor):
    mean_tensor = torch.mean(each_iteration_tensor,dim = 0 )
    variance_tensor = torch.var(each_iteration_tensor,dim = 0)
    error_bar = torch.sqrt(variance_tensor)
    plt.errorbar(np.arange(each_iteration_tensor.shape[1]), mean_tensor.numpy(),yerr = error_bar.numpy(), fmt='o-')
    plt.show()
    return

    