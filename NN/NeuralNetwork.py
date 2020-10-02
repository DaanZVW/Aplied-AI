import numpy as np
import copy
import random

random.seed( 0 )

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

    return normalised_data, data, names

def sigmoid( x ):
    """Return the sigmoid of x

    Args:
        x (float): Value you want sigmoided

    Returns:
        float: sigmoided value
    """
    return 1 / (1 + np.exp(-x))

def get_random_number():
    """Get random number between -1 and 1

    Returns:
        [type]: [description]
    """
    return random.uniform( -1, 1 )

class Input_Perceptron:
    def __init__( self, value=0 ):
        """Make an input perceptron

        Args:
            value (int, optional): Value of a_value. Defaults to 0.
        """
        self.a_value = value
        self.name = "Input"

class Perceptron:
    def __init__( self ):
        """Make a perceptron
        """
        self.child_perceptrons = []
        self.child_weights = []
        self.parent_perceptrons = []
        self.delta = 0
        self.bias = get_random_number()
        self.z_value = 0
        self.a_value = 0
        self.name = "Perceptron"

    def feed_forward( self ):
        """Feed forward this perceptron
        """
        # Get result of previous layer times the designated weights
        self.z_value = np.dot( [child.a_value for child in self.child_perceptrons], self.child_weights )

        # Sigmoid the z_value plus the bias and put this in a_value
        self.a_value = sigmoid( self.z_value + self.bias )

    def back_propagation_last_layer( self, correct_output ):
        """Do back propagation for the last layer (only)

        Args:
            correct_output (float): The value it should display
        """
        self.delta = sigmoid( self.z_value ) * (correct_output - self.a_value)

    def back_propagation( self ):
        """Do back propagation for the perceptron
        """
        # for child in self.child_perceptrons:
        #     # Filter out input childs
        #     if child.name != "Perceptron":
        #         continue
            
        #     sum_weighted_parents = 0
        #     for parent, parent_weight in zip( self.parent_perceptrons, self.child_weights ):
        #         print( "yash" )

        #         # Filter out input parents
        #         if parent.name != "Perceptron":
        #             continue

        #         child_in_parent_index = parent.child_perceptrons.index[child]
        #         print( child_in_parent_index )
            
        #     child.delta = sigmoid( child.z_value ) * sum_weighted_parents

        sum_weighted_parents = 0
        for parent in self.parent_perceptrons:
            # Filter out input parents
            if parent.name != "Perceptron":
                continue
            
            parent_weight = parent.child_weights[parent.child_perceptrons.index( self )]
            sum_weighted_parents += parent.delta * parent_weight
        
        self.delta = sigmoid( self.z_value ) * sum_weighted_parents

    def update_perceptron( self, learning_constant ):
        """Update the weight and bias of the perceptron

        Args:
            learning_constant (int): learning constant
        """
        tmp_child_weight = []
        for child, child_weight in zip( self.child_perceptrons, self.child_weights ):
            child_weight += learning_constant * self.delta * child.a_value
            tmp_child_weight.append( child_weight )

        self.child_weights = tmp_child_weight
        self.bias += learning_constant * self.delta


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
            self.perceptrons[index] = [Perceptron() for _ in range( hidden_layers[index-1] )]

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
                perceptron.child_weights = [get_random_number() for _ in range( len( self.perceptrons[layer_index] ) )]

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
        """Set a_value of the input nodes to the dataset

        Args:
            input_conf (list): list with all the nodes from top to bottom
        """
        for input_node, input_set in zip( self.perceptrons[0], input_conf ):
            input_node.a_value = input_set

    def get_output_nodes( self ):
        """Get the a_value of all the output nodes

        Returns:
            list: list containing all the a_values of the end nodes
        """
        output_nodes_output = []
        for output_node in self.perceptrons[-1]:
            output_nodes_output.append( output_node.a_value )
        return output_nodes_output

    def feed_forward( self ):
        """Feed forward all perceptrons in the Neural Network
        """
        for layer in self.perceptrons[1:]:
            for node in layer:
                if node.name != "Perceptron":
                    continue

                node.feed_forward()

    def back_propagation_last_layer( self, correct_output ):
        """Do the back propagation for the last layer

        Args:
            correct_output (list): list containing all the correct outputs from top to bottom
        """
        for node, correct_output_node in zip( self.perceptrons[-1], correct_output ):
            node.back_propagation_last_layer( correct_output_node )

    def back_propagation( self ):
        """Do the regular back propagation for the entire Neural Network
        """
        for layer in self.perceptrons[1:-1]:
            for node in layer:
                node.back_propagation()

    def update_perceptrons( self, learning_constant ):
        """Update all the weights and biases of the Neural Network

        Args:
            learning_constant (int): Learning constant for better training
        """
        for layer in self.perceptrons[1:]:
            for node in layer:
                node.update_perceptron( learning_constant )
                # print( "updated node" )

    def test_dataset( self, dataset, correct_output ):
        """Test the dataset with correct_output and return a percentage

        Args:
            dataset (list): list containing the dataset
            correct_output (list): list containing the correct_outputs

        Returns:
            float: Return a percentage of correct guessed.
        """
        output_list_raw = []
        output_list = []
        correct_awnsers = 0
        for dataset_data, correct_output_data in zip( dataset, correct_output ):

            self.set_input_nodes( dataset_data )
            self.feed_forward()

            output_nodes = [1 if node>0.5 else 0 for node in self.get_output_nodes()]
            if output_nodes == correct_output_data:
                correct_awnsers += 1

            output_list.append( output_nodes )
            output_list_raw.append( self.get_output_nodes() )
        
        # print( [[b.child_weights for b in a] for a in self.perceptrons[1:]] )
        # print( [[b.bias for b in a] for a in self.perceptrons[1:]] )

        return round((correct_awnsers / len(correct_output)) * 100, 2), output_list, output_list_raw

    def train( self, dataset, correct_output, learning_constant, iterations ):
        """Train the Neural Network

        Args:
            dataset (list): List containing all the data
            correct_output (list): list containing all the awnsers to the dataset
            learning_constant (int): Learning constant for better training
            iterations (int): Amount of iterations which the dataset is trained for
        """
        learning_constant_step = (learning_constant[0] - learning_constant[1]) / iterations

        for iteration_counter in range(iterations):
            print( "Iteration: {}".format( iteration_counter+1 ), end="" )
            # print( [[b.child_weights for b in a] for a in self.perceptrons[1:]] )

            for dataset_data, correct_output_data in zip( dataset, correct_output ):

                self.set_input_nodes( dataset_data )

                # print( dataset_data, correct_output_data )

                # print( [a.a_value for a in self.perceptrons[0]], dataset_data, correct_output_data )

                self.feed_forward()
                self.back_propagation_last_layer( correct_output_data )
                self.back_propagation()
                self.update_perceptrons( learning_constant[0] - learning_constant_step*iteration_counter )

            print( " ",self.test_dataset( dataset, correct_output )[0] )

if __name__ == "__main__":
    # print(  import_data_from_file( "iris.data" )[2] )

    john = Neural_Network( n_inputs=2, hidden_layers=[2], n_outputs=1 )
    john.connect_all()
    # john.print_network()

    dataset = [[0,0],[0,1],[1,0],[1,1]]
    correct_output = [[0],[1],[1],[0]]

    john.train( dataset=dataset, correct_output=correct_output, learning_constant=[0.85,0.005], iterations=10000 )
    john_resultaat = john.test_dataset( dataset=dataset, correct_output=correct_output )

    print( john_resultaat )