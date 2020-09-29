import numpy as np
import copy

def sigmoid( x ):
    return 1 / (1 + np.exp(-x))

class Input_Perceptron:
    def __init__( self, value=0 ):
        self.a_value = value
        self.name = "Input"

class Perceptron:
    def __init__( self ):
        self.child_perceptrons = []
        self.child_weights = []
        self.parent_perceptrons = []
        self.delta = 0
        self.bias = 0
        self.a_value = 0
        self.name = "Perceptron"

    def forward_propagation( self ):
        weighted_result = np.dot( [child.a_value for child in self.child_perceptrons], self.child_weights )
        return sigmoid( weighted_result + self.bias )

    def backward_propagation( self ):
        pass

    def sigma( self ):
        pass

    def update_neuron( self ):
        pass


class Neural_Network:
    def __init__( self, n_inputs, hidden_layers, n_outputs ):
        """Make neural network with given configuration

        Args:
            n_inputs (int): amount of input nodes
            hidden_layers (list): amount of nodes in layer given in list
            n_outputs (int): amount of output nodes
        """
        self.perceptrons = [[] for _ in range( len(hidden_layers) + 2 )]
        self.perceptrons[0] = [Input_Perceptron() for _ in range( n_inputs )]

        for index in range(1, len(hidden_layers)+1):
            self.perceptrons[index] = [Perceptron() for _ in range( hidden_layers[-1] )]

        self.perceptrons[-1] = [Perceptron() for _ in range( n_outputs )]
        self.inputs = self.perceptrons[0]

    def connect_all( self ):
        """This function will connect every node to eachother
        """
        for layer_index, layer in enumerate( self.perceptrons[1:] ):
            for perceptron in layer:
                # Take childeren from one layer back
                # NOTE: because we dont want layer 1 (inputs) we dont need index highering
                perceptron.child_perceptrons = self.perceptrons[layer_index]
                perceptron.child_weights = [0 for _ in range( len( self.perceptrons[layer_index] ) )]

                # Check if layer is not output layer
                # Make parents if true from one layer forward
                # NOTE: because we dont want layer 1 (inputs) we need to higher index by two
                if layer_index != len(self.perceptrons)-2:
                    perceptron.parent_perceptrons = self.perceptrons[layer_index+2]

    def print_network( self ):
        """Show the network in a useable form
        """   
        print("Layer: 0\n","\n".join( [ node.name for node in self.perceptrons[0] ] ), sep="")
        for layer_index, layer in enumerate(self.perceptrons[1:]):
            print("\nlayer: {}".format( layer_index+1 ))
            for node in layer:
                print( node.name, [ child.name for child in node.child_perceptrons ],
                       [parent.name for parent in node.parent_perceptrons], 
                       node.child_weights )

    def set_input_nodes( self, input_conf ):
        for input_node, input_set in zip( self.perceptrons[0], input_conf ):
            input_node.a_value = input_set

    def train( self, dataset, correct_output, iterations ):
        for iteration_counter in range(iterations):
            for dataset_data, correct_output_data in zip( dataset, correct_output ):

                self.set_input_nodes( dataset_data )
                
        

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

    john = Neural_Network( n_inputs=2, hidden_layers=[2], n_outputs=1 )
    john.connect_all()
    # john.print_network()

    dataset = [[0,0],[0,1],[1,0],[1,1]]
    correct_output = [0,1,1,0]

    john.train( dataset=dataset, correct_output=correct_output, iterations=1 )

    # a = [0,1,2,3,4,5,0]
    # b = [0,1,2,3,4,5,6]
    # print( [ap * bp for ap, bp in zip( a,b )] )
    # print( sum([ap * bp for ap, bp in zip( a,b )]) )
    # print( np.dot( a,b ) )