import torch
import numpy as np

class Model(torch.nn.Module):
    #NOTE we're actually using tanh to keep values bounded at the end, ReLu oonly fine for hidden layers
    def __init__(self, input_size, hidden_sizes, output_size, activation_fun=torch.nn.Tanh(), output_activation=None):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes #we keep this customizable for now, still testing some configs and also enables different
        # capacities for policy and critic networks
        self.output_size = output_size
        self.output_activation = output_activation

        #Now we build them up layer by layer
        layer_sizes = []
        layer_sizes.append(self.input_size)
        for hidden_dim in self.hidden_sizes:
            layer_sizes.append(hidden_dim)
        self.layers = torch.nn.ModuleList()
        for idx in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[idx]
            out_dim = layer_sizes[idx + 1]
            layer = torch.nn.Linear(in_dim, out_dim)
            self.layers.append(layer)

        self.activations = []
        for x in range(len(self.layers)):
            self.activations.append(activation_fun)
        final_layer_inputdim = self.hidden_sizes[-1]
        final_layer_outputdim = self.output_size
        self.output_layer = torch.nn.Linear(final_layer_inputdim, final_layer_outputdim)

    def forward(self, x):
        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            activation_function = self.activations[layer_idx]
            x = layer(x)
            x = activation_function(x)

        if self.output_activation is not None:
            output = self.output_layer(x)
            output = self.output_activation(output)
        else:
            output = self.output_layer(x)
        return output

    def predict(self, x):
        with torch.no_grad():
            #MAKE SURE to keep in float32
            input_tensor = torch.from_numpy(x.astype(np.float32))
            prediction = self.forward(input_tensor)
            output_numpy = prediction.numpy()
        return output_numpy
