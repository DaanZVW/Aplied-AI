import numpy as np
import copy

class Input_Neuron:
    def __init__( self ):
        pass

class Weighted_Neuron:
    def __init__( self, neuron, weight ):
        self.neuron = neuron
        self.weight = weight

class Neuron:
    def __init__( self ):
        self.neuron_childs = []
        self.neuron_parents = []
        self.delta = 0
        self.bias = 0

    def forward_propagation( self ):
        pass

    def backward_propagation( self ):
        pass

    def sigma( self ):
        pass

    def update_neuron( self ):
        pass

class Neural_network:
    def __init__(self, n_inputs, n_outputs, conf_neurons):
        self.neurons = []
    

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
    print(  import_data_from_file( "iris.data" )[2] )
    