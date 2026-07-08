import torch
from torch import nn 

class neural_net(nn.Module):
    """
    Defines a class for creating a dynamic neural net which can be tuned based on ones preference.
    
    Parameters
    ----------------------
    layer_geometry: torch 1D tensor of dtype torch.int defining the number of nodes in each layer
    activation_functions: torch.nn.ModuleList listing all the activation functions used at the end of each layer
    """
    def __init__(self, layer_geometry:torch.Tensor, activation_functions:nn.ModuleList ):
        super().__init__()

        # Makes checks about the inputs
        self.checks(layer_geometry,activation_functions)
        
        # Define the layer geometry in a modulelist
        self.layers = nn.ModuleList()
        for i in range(1,layer_geometry.shape[0]):
            self.layers.append(nn.Linear(in_features=layer_geometry[i-1],out_features=layer_geometry[i])) 
        
        # Define the activation function geometry
        self.activation_function = activation_functions
        return
    
    def checks(self,layer_geometry,activation_functions):
        """
        Function to take into account some basic checks for the layer_geometry and activation_functions
        """
        if layer_geometry.dtype !=torch.int :
            raise Exception("wrong dtype for layer_geometry")
        
        # Checks if both have same length
        if((len(layer_geometry)-1)!= len(activation_functions)):
            raise Exception("The length of the layer_geometry and activation_functions is not same")
        

    def forward(self,input_vector:torch.Tensor):
        """
        Forward function of the Neural net.
        """
        x = input_vector
        for i in range(len(self.layers)):
            x = self.activation_function[i](self.layers[i](x))
        return(x)

if __name__=="__main__":
    activation_functions = nn.ModuleList([nn.ReLU(),nn.Tanh()])
    layer_geometry = torch.tensor([4,64,2],dtype=torch.int)
    nn_object = neural_net(layer_geometry,activation_functions=activation_functions)
    print(nn_object)
    input_vector = [[1,2,3,4]]
    print(nn_object.forward(torch.tensor(input_vector,dtype=torch.float32)))
    print(input_vector)
    # print(nn_object.)