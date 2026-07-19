import os
import torch 
import pennylane as qp
import matplotlib.pyplot as plt
#################################
#################################
def one_layer_of_qvc(weight_matrix:torch.Tensor,n_wires):
    """
    Defines one layer of the quantum neural net

    Parameters
    ------------------------
    weight_matrix: torch tensor of size = (n_wires,3) due to the ansatz
    n_wires: number of wires in the circuit

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

#################################
#################################
def complete_variational_quantum_circuit_function(input_vector:torch.Tensor,
                                                complete_weight_matrix:torch.Tensor)->torch.Tensor:
    """
    Defines the complete circuit for the quantum neural net (except for the encoding).

    Parameters
    ------------------------------------------
    complete_weight_matrix: torch tensor of size (number_of_layer,number_of_wires,3)
    input_vector : The input vector to the layer (batch, number_of_wires)
    """
    number_of_layers = complete_weight_matrix.shape[0]
    number_of_wires = complete_weight_matrix.shape[1]
    
    for wire_number in range(number_of_wires):
        qp.RZ(phi=input_vector[..., wire_number] * torch.pi, wires=wire_number)

    qp.Barrier(wires = range(number_of_wires))

    for layer_number in range(number_of_layers):
        one_layer_of_qvc(complete_weight_matrix[layer_number,:,:],n_wires=complete_weight_matrix.shape[1])
        qp.Barrier(wires = range(number_of_wires))

    # Can generalize the measurement later
    final_array = [qp.expval(qp.Z(w)) for w in range(number_of_wires)]
    return final_array
#################################
#################################
def complete_variational_quantum_circuit_function_entanglement(input_vector:torch.Tensor,
                                                complete_weight_matrix:torch.Tensor,
                                                wire:int)->torch.Tensor:
    """
    Defines the complete circuit for the quantum neural net (except for the encoding), however, this is 
    mainly used for computing the entanglement entropy.

    Parameters
    -----------------------
    complete_weight_matrix: torch tensor of size (number_of_layer,number_of_wires,3)
    input_vector : The input vector
    """
    number_of_layers = complete_weight_matrix.shape[0]
    number_of_wires = complete_weight_matrix.shape[1]
    
    for wire_number in range(number_of_wires):
        qp.RZ(phi=input_vector[..., wire_number] * torch.pi, wires=wire_number)

    qp.Barrier(wires = range(number_of_wires))

    for layer_number in range(number_of_layers):
        one_layer_of_qvc(complete_weight_matrix[layer_number,:,:],n_wires=complete_weight_matrix.shape[1])
        qp.Barrier(wires = range(number_of_wires))

    return qp.vn_entropy(wires=[wire], log_base=2)
#################################
#################################
class complete_quantum_variational_circuit(torch.nn.Module):
    """
    Class which defines the complete quantum neural net. 

    Parameters
    --------------
    qvc: quantum function for the circuit of the quantum neural net
        ( should have the same inputs as complete_variational_quantum_circuit_function)
    
    complete_weight_matrix: torch tensor of size (number_of_layer,number_of_wires,3)
    
    dev: pennylane quantum device
    """
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
        self.weights = torch.nn.Parameter(complete_weight_matrix)

        self._device_name = dev.short_name if hasattr(dev, "short_name") else dev.name
        self._device_kwargs = {"wires":self.number_of_wires}

    def build_qnode(self):
        """
        Function to build a qnode for the quantum circuit. 
        This also takes into account the encoding of the input (Angle encoding).
        Note that the circuit function inside takes "input_vector" as an input.
        """

        @qp.qnode(device = self.device, interface="torch")
        def circuit(input_vector):
            # --- state preparation ---
            qp.AngleEmbedding(input_vector*torch.pi, wires=range(self.number_of_wires))

            qp.Barrier(wires=range(self.number_of_wires))
            # --- variational circuit ---
            final_result = self.qvc(input_vector = input_vector,
                                complete_weight_matrix= self.weights)
            return(final_result)
        
        return circuit

    def clone_complete_quantum_variational_circuit(self):
        new_complete_weight_matrix = torch.empty_like(self.weights)
        new_qvc = self.qvc
        new_dev = qp.device(self._device_name,**self._device_kwargs)
        return(complete_quantum_variational_circuit(qvc = new_qvc,complete_weight_matrix=new_complete_weight_matrix,dev = new_dev))

    def draw_quantum_circuit(self, input_vector:torch.Tensor):
        """
        Function used to draw the complete quantum circuit
        """
        circuit = self.build_qnode()

        print(qp.draw(circuit)(input_vector))
        
        qp.draw_mpl(qnode=circuit,style = "pennylane")(input_vector)
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        # plt.show()

        return 
    
    def forward(self, input_vector:torch.Tensor):
        """
        Forward function of the neural network
        """
        circuit = self.build_qnode()
        
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