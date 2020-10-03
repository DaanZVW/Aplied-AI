import numpy as np
import copy
import random
import time

# Set random seed to 0 for reproduction
random.seed( 0 )

# Functions for neural network which are is not fitting for in the classes 
def import_data_from_file( filename ):
    """Import data from filename (csv format)

    Args:
        filename (string): name of a csv file

    Returns:
        tuple: returns a the data normalised and the correct output
    """
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

    return normalised_data, names

def convert_names_to_correct_output( names ):
    """Convert names to a list with a 1 for true and 0 for false

    Args:
        names (list): list with different names

    Returns:
        list: returns a list 1 for true and 0 for false for all the names
    """
    unique_names = list(np.unique( names ))

    indexed_names = []
    for name in names:
        name_index = unique_names.index( name )
        indexed_names.append( [1 if name_index == unique_index else 0 for unique_index in range(len(unique_names))] )

    return indexed_names

def sigmoid( z ):
    """Return the sigmoid of z

    Args:
        z (float): Value you want sigmoided

    Returns:
        float: sigmoided value
    """
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative( z ):
    """Return the derivative of the sigmoid of z

    Args:
        z (float): Value you want sigmoided

    Returns:
        float: derivative sigmoided value
    """
    return sigmoid( z ) * (1 - sigmoid( z ))

def get_random_number():
    """Get random number between -1 and 1

    Returns:
        [type]: [description]
    """
    return random.uniform( -1, 1 )

# =================================================================
#                      Neuron classes
# =================================================================

class Input_Neuron:
    def __init__( self, value=0 ):
        """Make an input neuron

        Args:
            value (int, optional): Value of a_value. Defaults to 0.
        """
        self.a_value = value
        self.name = "Input"

class Neuron:
    def __init__( self ):
        """Make a neuron
        """
        self.child_neurons = []
        self.child_weights = []
        self.parent_neurons = []
        self.delta = 0
        self.bias = get_random_number()
        self.z_value = 0
        self.a_value = 0
        self.name = "Neuron"

    def feed_forward( self ):
        """Feed forward this neuron
        """
        # Get result of previous layer times the designated weights
        self.z_value = np.dot( [child.a_value for child in self.child_neurons], self.child_weights ) + self.bias

        # Sigmoid the z_value plus the bias and put this in a_value
        self.a_value = sigmoid( self.z_value )

    def back_propagation_last_layer( self, correct_output ):
        """Do back propagation for the last layer (only)

        Args:
            correct_output (float): The value it should display
        """
        self.delta = sigmoid_derivative( self.z_value ) * (correct_output - self.a_value)

    def back_propagation( self ):
        """Do back propagation for the neuron
        """
        sum_weighted_parents = 0
        for parent in self.parent_neurons:
            # Filter out input parents
            if parent.name != "Neuron":
                continue
            
            parent_weight = parent.child_weights[parent.child_neurons.index( self )]
            sum_weighted_parents += parent.delta * parent_weight
        
        self.delta = sigmoid_derivative( self.z_value ) * sum_weighted_parents

    def update_neuron( self, learning_constant ):
        """Update the weight and bias of the neuron

        Args:
            learning_constant (int): learning constant
        """
        tmp_child_weight = []
        for child, child_weight in zip( self.child_neurons, self.child_weights ):
            child_weight += learning_constant * self.delta * child.a_value
            tmp_child_weight.append( child_weight )

        self.child_weights = tmp_child_weight
        self.bias += learning_constant * self.delta

# =================================================================
#                           NN class
# =================================================================

class Neural_Network:
    def __init__( self, n_inputs, hidden_layers, n_outputs ):
        """Make neural network with given configuration

        Args:
            n_inputs (int): amount of input nodes
            hidden_layers (list): amount of nodes in layer given in list
            n_outputs (int): amount of output nodes
        """
        self.neurons = [[] for _ in range( len(hidden_layers) + 2 )]
        self.neurons[0] = [Input_Neuron() for _ in range( n_inputs )]

        for index in range(1, len(hidden_layers)+1):
            self.neurons[index] = [Neuron() for _ in range( hidden_layers[index-1] )]

        self.neurons[-1] = [Neuron() for _ in range( n_outputs )]
        self.inputs = self.neurons[0]

    def connect_all( self ):
        """This function will connect every node to eachother
        """
        for layer_index, layer in enumerate( self.neurons[1:] ):
            for neuron in layer:
                # Take childeren from one layer back
                # NOTE: because we dont want layer 1 (inputs) we dont need index highering
                neuron.child_neurons = self.neurons[layer_index]
                neuron.child_weights = [get_random_number() for _ in range( len( self.neurons[layer_index] ) )]

                # Check if layer is not output layer
                # Make parents if true from one layer forward
                # NOTE: because we dont want layer 1 (inputs) we need to higher index by two
                if layer_index != len(self.neurons)-2:
                    neuron.parent_neurons = self.neurons[layer_index+2]

    def print_network( self ):
        """Show the network in a useable form
        """   
        print("Layer: 0\n","\n".join( [ node.name for node in self.neurons[0] ] ), sep="")
        for layer_index, layer in enumerate(self.neurons[1:]):
            print("\nlayer: {}".format( layer_index+1 ))
            for node in layer:
                print( node.name, [ child.name for child in node.child_neurons ],
                       [parent.name for parent in node.parent_neurons], 
                       node.child_weights )

    def set_input_nodes( self, input_conf ):
        """Set a_value of the input nodes to the dataset

        Args:
            input_conf (list): list with all the nodes from top to bottom
        """
        for input_node, input_set in zip( self.neurons[0], input_conf ):
            input_node.a_value = input_set

    def get_output_nodes( self ):
        """Get the a_value of all the output nodes

        Returns:
            list: list containing all the a_values of the end nodes
        """
        output_nodes_output = []
        for output_node in self.neurons[-1]:
            output_nodes_output.append( output_node.a_value )
        return output_nodes_output

    def feed_forward( self ):
        """Feed forward all neurons in the Neural Network
        """
        for layer in self.neurons[1:]:
            for node in layer:
                if node.name != "Neuron":
                    continue

                node.feed_forward()

    def back_propagation_last_layer( self, correct_output ):
        """Do the back propagation for the last layer

        Args:
            correct_output (list): list containing all the correct outputs from top to bottom
        """
        for node, correct_output_node in zip( self.neurons[-1], correct_output ):
            node.back_propagation_last_layer( correct_output_node )

    def back_propagation( self ):
        """Do the regular back propagation for the entire Neural Network
        """
        for layer in reversed(self.neurons[1:-1]):
            for node in layer:
                node.back_propagation()

    def update_neurons( self, learning_constant ):
        """Update all the weights and biases of the Neural Network

        Args:
            learning_constant (int): Learning constant for better training
        """
        for layer in self.neurons[1:]:
            for node in layer:
                node.update_neuron( learning_constant )

    def test_dataset( self, dataset, correct_output ):
        """Test the dataset with correct_output and return a percentage

        Args:
            dataset (list): list containing the dataset
            correct_output (list): list containing the correct_outputs

        Returns:
            float: Return a percentage of correct guessed.
        """
        wrong_factor = 0
        for dataset_data, correct_output_data in zip( dataset, correct_output ):

            self.set_input_nodes( dataset_data )
            self.feed_forward()
            for correct_output_datapoint, node_output in zip( correct_output_data, self.get_output_nodes() ):
                wrong_factor += abs(correct_output_datapoint - node_output)

        return wrong_factor

    def get_percentage_correctness( self, dataset, correct_output ):
        """Test the current Neural Network and give back percentages

        Args:
            dataset (list): list with the dataset
            correct_output (list): list with the correct awnsers of the dataset

        Returns:
            tuple: percentage, output_list_raw
                percentage: percentage of correctness
                output_list_raw: give back the raw anwser of the Neural Network
        """
        output_list_raw = []
        correct_awnsers = 0
        for dataset_data, correct_output_data in zip( dataset, correct_output ):

            self.set_input_nodes( dataset_data )
            self.feed_forward()

            output_nodes = [1 if node>0.5 else 0 for node in self.get_output_nodes()]
            if output_nodes == correct_output_data:
                correct_awnsers += 1

            output_list_raw.append( [round(np.float(x), 5) for x in self.get_output_nodes()] )

        return round((correct_awnsers / len(correct_output)) * 100, 2), output_list_raw

    def train( self, dataset, correct_output, learning_constant, iterations, show_info=True ):
        """Train the Neural Network

        Args:
            dataset (list): List containing all the data
            correct_output (list): list containing all the awnsers to the dataset
            learning_constant (list): Learning constant for better training
            iterations (int): Amount of iterations which the dataset is trained for
        
        NOTE: The learning constant consist out of 2 variables. index 0 is the initial
        learning constant. When the error_margin highers it will be multiplied by index
        1. So index 1 has to be below 1.0 to get a valiable training.
        """
        test_results = []
        tmp_learning_constant = learning_constant[0]

        for iteration_counter in range(iterations):
            for dataset_data, correct_output_data in zip( dataset, correct_output ):

                self.set_input_nodes( dataset_data )
                self.feed_forward()
                self.back_propagation_last_layer( correct_output_data )
                self.back_propagation()
                self.update_neurons( tmp_learning_constant )

            test_data = self.test_dataset( dataset, correct_output )
            test_results.append(test_data)

            # Decrease learning constant by given factor if error gets higher
            if len(test_results) >= 2:
                if test_data > test_results[-2]:
                    tmp_learning_constant *= learning_constant[1]              

            if show_info:
                print( "Iteration: {} {} {}".format( iteration_counter+1, round(test_data, 5), tmp_learning_constant))

        return test_results

# =================================================================
#                          Main function
# =================================================================

if __name__ == "__main__":

    # # XOR
    XOR = Neural_Network( n_inputs=2, hidden_layers=[2], n_outputs=1 )
    XOR.connect_all()

    dataset = [[0,0],[0,1],[1,0],[1,1]]
    correct_output = [[0],[1],[1],[0]]

    XOR.train( dataset=dataset, correct_output=correct_output, learning_constant=[10,0.8], iterations=300, show_info=False )
    XOR_result = XOR.get_percentage_correctness( dataset=dataset, correct_output=correct_output )

    print( "XOR:        {}".format(XOR_result) )
    
    # # Full Adder
    Full_Adder = Neural_Network( n_inputs=3, hidden_layers=[3], n_outputs=2 )
    Full_Adder.connect_all()

    dataset =        [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    correct_output = [[0,0],  [0,1],  [0,1],  [1,0],  [0,1],  [1,0],  [1,0],  [1,1]]

    Full_Adder.train( dataset=dataset, correct_output=correct_output, learning_constant=[10,0.9], iterations=200, show_info=False )
    Full_Adder_result = Full_Adder.get_percentage_correctness( dataset=dataset, correct_output=correct_output )

    print( "Full_adder: {}".format(Full_Adder_result) )

    # Iris Dataset
    Iris = Neural_Network( n_inputs=4, hidden_layers=[5], n_outputs=3 )
    Iris.connect_all()

    Iris_dataset = import_data_from_file( "iris.data" )
    dataset = Iris_dataset[0]
    correct_output = convert_names_to_correct_output( Iris_dataset[1] )

    begin_time = time.time_ns()
    Iris.train( dataset=dataset, correct_output=correct_output, learning_constant=[10,0.4], iterations=20000, show_info=True )
    print( "Took: {} seconds".format( round(((time.time_ns() - begin_time)/1000000000), 2) ))
    Iris_result = Iris.get_percentage_correctness( dataset=dataset, correct_output=correct_output )

    print( "Iris:       {}".format(Iris_result) )

    # john_res = john.train( dataset=dataset, correct_output=correct_output, learning_constant=[10,0.5], iterations=500, show_info=False )
    # john_resultaat = john.get_percentage_correctness( dataset=dataset, correct_output=correct_output )

    # print( "Iteration {} was lowest with correct factor of {}".format( john_res.index(min(john_res))+1, min(john_res) ))
    # print( john_resultaat )