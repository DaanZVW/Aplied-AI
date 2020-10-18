import random
import copy
from functools import reduce
from operator import add

random.seed( 0 )

def individual (length , min, max):
    """
    Creates an individual for a population
    :param length: the number of values in the list
    :param min: the minimum value in the list of values
    :param max: the maximal value in the list of values
    :return:
    """

def population (count , length , min, max):
    """
    Create a number of individuals (i.e., a population).
    :param count: the desired size of the population
    :param length: the number of values per individual
    :param min: the minimum in the individual’s values
    :param max: the maximal in the individual’s values
    """

def fitness(individual):
    """
    Determine the fitness of an individual. Lower is better.
    :param individual: the individual to evaluate
    I removed target because it was not nessesary in this exercise
    """

def grade(population ):
    """
    Find average fitness for a population
    :param population: population to evaluate
    I removed target because it was not nessesary in this exercise
    """

def evolve(population, retain = 0.2, random_select = 0.05, mutate = 0.01):
    """
    Function for evolving a population , that is, creating
    offspring (next generation population) from combining
    (crossover) the fittest individuals of the current
    population
    :param population: the current population
    :param target: the value that we are aiming for
    :param retain: the portion of the population that we allow to spawn offspring
    :param random_select: the portion of individuals that are selected at random.random , not based on their score
    :param mutate: the amount of random.random change we apply to new offspring
    :return: next generation population
    """

def find_best_solution():
    """Brute force the best solution for exercise 6.2

    Returns:
        tuple: tuple containing the configuration of invididual and the outcome
    """                    

if __name__ == "__main__":
    # Set some variables for the evolve function
    p_count = 1000 # number of individuals in population
    i_length = 4 # N
    i_min = 0 # value range for generating individuals
    i_max = 63

    # Generate a population with the just set variables
    p = population(p_count , i_length , i_min , i_max)

    # Make highest population
    p_highest = (copy.deepcopy(p), 0)

    # Make the fitness history and fill it with the grade function
    fitness_history = [grade(p)]
    
    # Print max grade in population at start
    print( "Start:", *fitness_history )
    
    # Evolve the algorithm for 100 iterations
    for _ in range(30):
        p = evolve( p, retain=0.4, random_select=0.0, mutate=0.3 )
        score = grade(p)
        fitness_history.append(score)

        if ( fitness_history[-1] > fitness_history[p_highest[1]] ):
            p_highest = (copy.deepcopy(p), len(fitness_history)-1)
    
    # Print the whole fitness history
    print( "Finish:", fitness_history[-1])

    # Print the highest graded population
    print( "Population index: {}\nPopulation: {}".format(p_highest[1], p_highest[0]), sep="\n\n" )

    # Print the best posible answer for our problem
    print(find_best_solution())

    # NOTE: The Evolutionary Algorithm is way faster than brute forcing the solution
