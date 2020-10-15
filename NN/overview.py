import numpy as np
import copy
import random
import time

# Set random seed to 0 for reproduction
random.seed( 0 )

# Functions for neural network which are is not fitting for in the classes 
def import_data_from_file( filename, delim, data_cols, name_cols ):
    """Import data from filename (csv format)

    Args:
        filename (string): name of a csv file

    Returns:
        tuple: returns a the data normalised and the correct output
    """

def convert_names_to_correct_output( names ):
    """Convert names to a list with a 1 for true and 0 for false

    Args:
        names (list): list with different names

    Returns:
        list: returns a list 1 for true and 0 for false for all the names
    """
    
def sigmoid( z ):
    """Return the sigmoid of z

    Args:
        z (float): Value you want sigmoided

    Returns:
        float: sigmoided value
    """

def sigmoid_derivative( z ):
    """Return the derivative of the sigmoid of z

    Args:
        z (float): Value you want sigmoided

    Returns:
        float: derivative sigmoided value
    """

def get_random_number():
    """Get random number between -1 and 1

    Returns:
        [type]: [description]
    """

# =================================================================
#                      Neuron classes
# =================================================================

class Input_Neuron:
    """An input neuron which is used in the Neural Network
    """  
    def __init__( self, value=0 ):
        """Make an input neuron

        Args:
            value (int, optional): Value of a_value. Defaults to 0.
        """

class Neuron:
    """A neuron which is used in the Neural Network
    """
    def __init__( self ):
        """Make a neuron
        """

    def feed_forward( self ):
        """Feed forward this neuron
        """

    def back_propagation_last_layer( self, correct_output ):
        """Do back propagation for the last layer (only)

        Args:
            correct_output (float): The value it should display
        """

    def back_propagation( self ):
        """Do back propagation for the neuron
        """

    def update_neuron( self, learning_constant ):
        """Update the weight and bias of the neuron

        Args:
            learning_constant (int): learning constant
        """

# =================================================================
#                           NN class
# =================================================================

class Neural_Network:
    """Interface class for interacting with the Neurons
    """

    def __init__( self, n_inputs, hidden_layers, n_outputs ):
        """Make neural network with given configuration

        Args:
            n_inputs (int): amount of input nodes
            hidden_layers (list): amount of nodes in layer given in list
            n_outputs (int): amount of output nodes
        """

    def connect_all( self ):
        """This function will connect every node to eachother
        """

    def print_network( self ):
        """Show the network in a useable form
        """   

    def set_input_nodes( self, input_conf ):
        """Set a_value of the input nodes to the dataset

        Args:
            input_conf (list): list with all the nodes from top to bottom
        """

    def feed_forward( self ):
        """Feed forward all neurons in the Neural Network
        """

    def back_propagation_last_layer( self, correct_output ):
        """Do the back propagation for the last layer

        Args:
            correct_output (list): list containing all the correct outputs from top to bottom
        """

    def back_propagation( self ):
        """Do the regular back propagation for the entire Neural Network
        """

    def update_neurons( self, learning_constant ):
        """Update all the weights and biases of the Neural Network

        Args:
            learning_constant (int): Learning constant for better training
        """

    def test_dataset( self, dataset, correct_output ):
        """Test the dataset with correct_output and return a percentage

        Args:
            dataset (list): list containing the dataset
            correct_output (list): list containing the correct_outputs

        Returns:
            float: Return a percentage of correct guessed.
        """

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

# =================================================================
#                          Main function
# =================================================================

if __name__ == "__main__":
    # ===
    # XOR
    # ===
    XOR = Neural_Network( n_inputs=2, hidden_layers=[2], n_outputs=1 )
    XOR.connect_all()

    dataset = [[0,0],[0,1],[1,0],[1,1]]
    correct_output = [[0],[1],[1],[0]]

    XOR.train( dataset=dataset, correct_output=correct_output, learning_constant=[10,0.8], iterations=300, show_info=False )
    XOR_result = XOR.get_percentage_correctness( dataset=dataset, correct_output=correct_output )

    print( "XOR:        {}".format(XOR_result) )

    # ========== 
    # Full Adder
    # ==========
    Full_Adder = Neural_Network( n_inputs=3, hidden_layers=[3], n_outputs=2 )
    Full_Adder.connect_all()

    dataset =        [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    correct_output = [[0,0],  [0,1],  [0,1],  [1,0],  [0,1],  [1,0],  [1,0],  [1,1]]

    Full_Adder.train( dataset=dataset, correct_output=correct_output, learning_constant=[10,0.9], iterations=200, show_info=False )
    Full_Adder_result = Full_Adder.get_percentage_correctness( dataset=dataset, correct_output=correct_output )

    print( "Full_adder: {}".format(Full_Adder_result) )

    # ============
    # Iris Dataset
    # ============
    # Excelent results, but expensive (computional) to train
    # Iris = Neural_Network( n_inputs=4, hidden_layers=[5,4], n_outputs=3 )

    # Above average results, fast to train
    Iris = Neural_Network( n_inputs=4, hidden_layers=[4], n_outputs=3 )
    
    Iris.connect_all()

    Iris_dataset = import_data_from_file( filename="iris.data", delim=",", data_cols=[0,1,2,3], name_cols=[4] )
    dataset = Iris_dataset[0]
    correct_output = convert_names_to_correct_output( Iris_dataset[1] )

    # Excelent results, but expensive (computional) to train
    # Iris.train( dataset=dataset, correct_output=correct_output, learning_constant=[10,0.38], iterations=3000, show_info=False )

    # Above average results, fast to train
    Iris.train( dataset=dataset, correct_output=correct_output, learning_constant=[10,0.6], iterations=1500, show_info=False )

    Iris_result = Iris.get_percentage_correctness( dataset=dataset, correct_output=correct_output )
    print( "Iris:       {}".format(Iris_result) )