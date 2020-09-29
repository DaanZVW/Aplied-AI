import numpy as np
import copy

def sigmoid( self, x ):
    return 1 / (1 + np.exp(-x))

class Input_Perceptron:
    def __init__( self, value=0 ):
        self.value = value

    def set_value( self, value ):
        self.value = value

    def get_value( self ):
        return self.value

class Perceptron:
    def __init__( self ):
        self.perceptron_childs = []
        self.perceptron_parents = []
        self.delta = 0
        self.bias = 0

    def add_childs( self, childs ):
        self.perceptron_childs = childs
    
    def add_parents( self, parents ):
        self.perceptron_parents = parents

    def forward_propagation( self ):
        pass

    def backward_propagation( self ):
        pass

    def sigma( self ):
        pass

    def update_neuron( self ):
        pass

class Neural_Network:
    def __init__( self, n_inputs, conf_neurons, n_outputs ):
        self.perceptrons = [[] for _ in range( len(conf_neurons) + 2 )]
        self.perceptrons[0] = [Input_Perceptron() for _ in range( n_inputs )]

        for index in range(1, len(conf_neurons)+1):
            self.perceptrons[index] = [Perceptron() for _ in range( conf_neurons[-1] )]

        self.perceptrons[-1] = [Perceptron() for _ in range( n_outputs )]

        self.inputs = self.perceptrons[0]

    def connect_all( self ):
        for layer_index, layer in enumerate( self.perceptrons[1:] ):
            for perceptron in layer:
                # Check if layer is not input layer
                # Make childeren if true
                if layer_index != 0:
                    perceptron.add_childs( self.perceptrons[layer_index+1] )

                # Check if layer is not output layer
                # Make parents if true
                if layer_index != len(self.perceptrons)-2:
                    perceptron.add_parents( self.perceptrons[layer_index-1] )


def import_data_from_file( filename ):
    data = np.genfromtxt(
        filename,
        delimiter=',',
        usecols=[0,1,2,3],
        converters={
            5: lambda s: 0 if s == b"-1" else float(s),
            7: lambda s: 0 if s == b"-1" else float(s)
        })

    names = np.genfromtxt(filename, dtype=str, delimiter=',', usecols=[4])

    # Make copy of first row for column min and max list
    min_list = copy.deepcopy(data[0])
    max_list = copy.deepcopy(data[0])

    # Go through all the unseen data rows and check if it is
    # higher or lower than the found min or max of that col
    for row in data[1:]:
        for colindex in range(len(row)):
            min_list[colindex] = min( row[colindex], min_list[colindex] )
            max_list[colindex] = max( row[colindex], max_list[colindex] )

    # Make new data list with all the normalised data
    normalised_data = []

    # Append data with data_points
    for row in data:
        tmp_data = []

        # Rescale all the data training data with min, max normalisation
        for col, min_value, max_value in zip(row, min_list, max_list):
            tmp_data.append( ((col - min_value) / (max_value - min_value)) )
        
        normalised_data.append( tmp_data )

    return normalised_data, data, names


if __name__ == "__main__":
    # print(  import_data_from_file( "iris.data" )[2] )

    john = Neural_Network( 2, [2], 1 )
    john.connect_all()
    
    for i in john.perceptrons[1:]:
        print()
        for j in i:
            print( j.name, j.perceptron_childs, j.perceptron_parents )