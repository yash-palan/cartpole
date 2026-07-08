import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import class_defns.quantum_circuit_classes as qcc
from class_defns.utility_functions import loading_dot_pt_files
import pennylane as qp

def complete_variational_quantum_circuit_function(complete_weight_matrix:torch.Tensor)->torch.Tensor:
    """
    complete_weight_matrix: torch tensor of size (number_of_layer,number_of_wires,3)
    input_vector : The input vector
    """
    number_of_layers = complete_weight_matrix.shape[0]
    number_of_wires = complete_weight_matrix.shape[1]
    
    # for wire_number in range(number_of_wires):
    #     # qp.RZ(phi=input_vector[wire_number]*torch.pi,wires=wire_number)
    #     # Needs a little more understanding
    #     qp.RZ(phi=input_vector[..., wire_number] * torch.pi, wires=wire_number)

    # qp.Barrier(wires = range(number_of_wires))

    for layer_number in range(number_of_layers):
        qcc.one_layer_of_qvc(complete_weight_matrix[layer_number,:,:],n_wires=complete_weight_matrix.shape[1])
        qp.Barrier(wires = range(number_of_wires))

    # Can generalize the measurement later
    final_array = [qp.expval(qp.Z(w)) for w in range(number_of_wires)]
    return final_array

dev = qp.device("default.qubit")
base_path = os.getcwd()
complete_path = base_path+'/results/ansatz 2 qubit 3/'
# complete_path = base_path+'/results/ansatz 1 qubit 3/'
# complete_path = base_path+'/results/ansatz 2 qubit 2/'
# filename = "ansatz_1.pt"
# filename = "ansatz_1_seed_4.pt"
filename = "ansatz_1_seed_3.pt"

network = loading_dot_pt_files(complete_path+filename)
network.eval()
complete_weight_matrix = network.quantum_variational_circuit.weights
trained_circuit= qp.QNode(complete_variational_quantum_circuit_function ,device=dev, interface="torch")

# Decompose into Clifford+T
epsilon = 1e-3  # your choice of approximation precision — report this!
clifford_t_qnode = qp.clifford_t_decomposition(trained_circuit, epsilon=epsilon)

trained_params = complete_weight_matrix  # your optimized parameters
# clifford_t_qnode
with qp.Tracker(dev) as tracker:
    clifford_t_qnode(trained_params)

resources_lst = tracker.history["resources"]
print(resources_lst[0])

# Run it once to build the tape (needed to inspect gates)
# tape = qp.workflow.construct_tape(clifford_t_qnode)(trained_params)

# from collections import Counter

# gate_counts = Counter(op.name for op in tape.operations)
# t_count = gate_counts.get("T", 0) + gate_counts.get("Adjoint(T)", 0)
# print(gate_counts)
# print("T-count:", t_count)

# # Build a DAG-based depth count restricted to T/T-dagger gates
# from pennylane.tape import QuantumScript

# def t_depth(tape):
#     # crude layering: greedily assign each op to the earliest layer where its wires are free
#     wire_available_at = {w: 0 for w in tape.wires}
#     max_depth = 0
#     count = 0
#     for op in tape.operations:
#         # print(count)
#         # if(count == 1997):
#         #     print("hello")
#         if(len(op.wires)==0):
#             continue
#         start = max(wire_available_at[w] for w in op.wires)
#         is_t = op.name in ("T", "Adjoint(T)")
#         end = start + 1
#         for w in op.wires:
#             wire_available_at[w] = end
#         if is_t:
#             max_depth = max(max_depth, end)
#         count = count+1
#     return max_depth

# print("T-depth (upper bound):", t_depth(tape))