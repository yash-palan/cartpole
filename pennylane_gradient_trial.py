import pennylane as qp
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
# import numpy as np
import pennylane.numpy as np
import matplotlib.pyplot as plt

n_wires = 4
dev = qp.device('default.qubit',wires = n_wires)

def EntanglerCircuit(weights:np.array,wires:np.array ):
    if(len(weights.shape)!=2):
        raise Exception("issue with the input weights size")

    for i in range(len(weights)):
        for j in range(len(wires)):
            qp.RX(phi = weights[i,j],wires = j )

        qp.CNOT(wires =  (0,1))
        qp.CNOT(wires =  (1,2))
        qp.CNOT(wires =  (2,3))
        qp.CNOT(wires =  (3,0))
        qp.Barrier(wires=wires)


def circuit(weights, wires):
    EntanglerCircuit(weights,wires)
    return(qp.expval(qp.PauliZ(0)))


# q_node = qp.QNode(circuit,device = dev)
# q_node = qp.QNode(circuit,device = dev, interface="autograd", diff_method="parameter-shift")

q_node = qp.QNode(circuit,device= dev, interface="torch", diff_method="parameter-shift")


# weights = np.array([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7, 0.8]])
# weights = np.array([[0.1,0.2,0.3,0.4]], requires_grad = True)
weights = torch.tensor([[0.1,0.2,0.3,0.4]], requires_grad = True)

# print(weights.shape)
# print(len(shape))

print(qp.draw(qnode=q_node)(weights,wires = [0,1,2,3]))
qp.draw_mpl(qnode=q_node,style = "pennylane")(weights,wires = [0,1,2,3])
plt.show()

# expval = q_node(weights, wires=[0,1,2,3])

# print(expval)
# print(qp.jacobian(q_node)(weights,wires=[0,1,2,3]))
# loss = (1 - q_node(weights,wires = [0,1,2,3]))**2
loss = q_node(weights,wires=[0,1,2,3])
loss.backward()

print(weights.grad)
