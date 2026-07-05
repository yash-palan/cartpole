import os
import torch 
import pennylane as qp
import matplotlib.pyplot as plt

def one_layer_of_qvc(weight_matrix:torch.Tensor,n_wires):
    """
    weight_matrix: torch tensor of size = (n_wires,3) due to the ansatz
    """
    if(len(weight_matrix.shape)!=2):
        raise Exception("shape of the weight matrix is incorrect")
    if(weight_matrix.shape[0]!=n_wires):
        raise Exception("first shape of the weight matrix is not the same as the number of wires")
    if(weight_matrix.shape[1] !=3):
        raise Exception("You need Rx Ry and Rz for one layer. Check you matrix again")

    for wire in range(n_wires-1):
        qp.CNOT(wires=(wire,wire+1))

    for wire in range(n_wires):
        qp.RX(phi = weight_matrix[wire,0],wires = wire )
    
    for wire in range(n_wires):
        qp.RY(phi = weight_matrix[wire,1],wires = wire )

    for wire in range(n_wires):
        qp.RZ(phi = weight_matrix[wire,2],wires = wire )

    return 

# def measurements():
#     return(qp.expval(qp.PauliZ(0),qp.PauliZ(1)))

def complete_variational_quantum_circuit_function(input_vector:torch.Tensor,
                                                complete_weight_matrix:torch.Tensor)->torch.Tensor:
    """
    complete_weight_matrix: torch tensor of size (number_of_layer,number_of_wires,3)
    input_vector : The input vector
    """
    number_of_layers = complete_weight_matrix.shape[0]
    number_of_wires = complete_weight_matrix.shape[1]
    
    for wire_number in range(number_of_wires):
        # qp.RZ(phi=input_vector[wire_number]*torch.pi,wires=wire_number)
        # Needs a little more understanding
        qp.RZ(phi=input_vector[..., wire_number] * torch.pi, wires=wire_number)

    qp.Barrier(wires = range(number_of_wires))

    for layer_number in range(number_of_layers):
        one_layer_of_qvc(complete_weight_matrix[layer_number,:,:],n_wires=complete_weight_matrix.shape[1])
        qp.Barrier(wires = range(number_of_wires))

    # Can generalize the measurement later
    final_array = [qp.expval(qp.Z(w)) for w in range(number_of_wires)]
    return final_array


class complete_quantum_variational_circuit(torch.nn.Module):
    def __init__(self,qvc,complete_weight_matrix:torch.Tensor,dev):
        """
        complete_weight_matrix: torch tensor of size (number_of_layer,number_of_wires,3)
        qvc: quantum function ( should have the same inputs as complete_variational_quantum_circuit_function)
        """
        super().__init__()
        self.qvc = qvc
        self.device = dev
        self.number_of_layers = complete_weight_matrix.shape[0]
        self.number_of_wires =  complete_weight_matrix.shape[1]
        # self.weights = complete_weight_matrix
        self.weights = torch.nn.Parameter(complete_weight_matrix)

    def build_qnode(self):
        # dev = 
        @qp.qnode(device = self.device, interface="torch")
        def circuit(input_vector):
            # --- state preparation ---
            qp.AngleEmbedding(input_vector*torch.pi, wires=range(self.number_of_wires))
            # trial_input_vector = input_vector*torch.pi
            # qp.AngleEmbedding(trial_input_vector, wires=range(self.number_of_wires))

            qp.Barrier(wires=range(self.number_of_wires))
            # --- variational circuit ---
            # return(self.qvc(input_vector = angle_embeddings,
            #                 complete_weight_matrix= self.weights,
            #                 number_of_layer=self.number_of_layers))
            final_result = self.qvc(input_vector = input_vector,
                                complete_weight_matrix= self.weights)
            return(final_result)
        
        return circuit

    def draw_quantum_circuit(self, input_vector:torch.Tensor):
        circuit = self.build_qnode()

        print(qp.draw(circuit)(input_vector))
        
        qp.draw_mpl(qnode=circuit,style = "pennylane")(input_vector)
        plt.show()

        return 
    
    def forward(self, input_vector:torch.Tensor):
        circuit = self.build_qnode()
        # print(qp.draw(circuit)(input_vector))
        
        # We need to concatenate so that the computational graph is not broken
        # Also, the dtype is wrong (it is float64) which we need to convert to 
        # float 32. The dim=1 takes care of the minibatch case 
        return (torch.stack(circuit(input_vector),dim=1).to(dtype=torch.float32))
    

if __name__=="__main__":
    # Check for working code
    complete_weight_matrix = torch.tensor([[[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9],[1.1,1.2,1.3]]]
                                          ,dtype=torch.float32)
    # input_vector = torch.tensor([[0.1],[0.2],[0.3],[0.4]],dtype=torch.float32)
    # input_vector = torch.tensor([[0.1,0.2,0.3,0.4]],dtype=torch.float32)
    input_vector = torch.tensor([[0.1,0.2,0.3,0.4],
                                 [0.1,0.2,0.3,0.4]],dtype=torch.float32)

    dev = qp.device('default.qubit',wires = complete_weight_matrix.shape[1])
    qvc_func = complete_variational_quantum_circuit_function


    qvc_object = complete_quantum_variational_circuit(qvc_func,complete_weight_matrix,dev)
    result = qvc_object.forward(input_vector)
    loss = result[0]
    print(result)
    # print(torch.stack(result,dim = 1))
    # CHeck for gradient flow
    # print(loss)
    
    # print(loss.backward())

    # print(qvc_object.weights.grad)