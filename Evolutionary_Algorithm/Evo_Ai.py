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
    return [ random.randint(min, max) for x in range(int(length)) ]

def population (count , length , min, max):
    """
    Create a number of individuals (i.e., a population).
    :param count: the desired size of the population
    :param length: the number of values per individual
    :param min: the minimum in the individual’s values
    :param max: the maximal in the individual’s values
    """
    return [ individual(length , min, max) for x in range(int(count)) ]

def fitness(individual):
    """
    Determine the fitness of an individual. Lower is better.
    :param individual: the individual to evaluate
    I removed target because it was not nessesary in this exercise
    """
    A, B, C, D = individual
    return (A-B)**2 + (C+D)**2 - (A-30)**3 - (C-40)**3

def grade(population ):
    """
    Find average fitness for a population
    :param population: population to evaluate
    I removed target because it was not nessesary in this exercise
    """
    summed = reduce(add , (fitness(x) for x in population ), 0)
    return summed / len( population )

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
    # Sort the population on fitness but only get the population in graded
    graded = [ (fitness(x), x) for x in population ]
    graded = [ x[1] for x in sorted(graded, reverse=True) ]

    # Retain the retain percentage of invidiuals
    retain_length = int(len(graded ) * retain)
    parents = graded [: retain_length ]
    
    # randomly add other individuals to promote genetic diversity
    for individual in graded[ retain_length :]:
        if random_select > random.random ():
            parents.append( individual )
    
    # create childeren for lost population in previous step
    desired_length = len( population ) - len( parents )
    children = []
    while len(children) < desired_length :
        male = random.randint(0, len(parents)-1)
        female = random.randint(0, len(parents)-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = int(len(male) / 2)
            child = male [: half] + female[half :]
            children.append(child)
    
    # mutate some of the childerens which are in the list
    for individual in children:
        if mutate > random.random ():
            pos_to_mutate = random.randint (0, len( individual)-1)
            # this mutation is not ideal , because it
            # restricts the range of possible values ,
            # but the function is unaware of the min/max
            # values used to create the individuals
            individual [ pos_to_mutate ] = random.randint(min( individual ), max( individual ))
    
    # put childeren in the parents list and return the new population
    parents.extend(children)
    return parents

def find_best_solution():
    """Brute force the best solution for exercise 6.2

    Returns:
        tuple: tuple containing the configuration of invididual and the outcome
    """
    highest_grade = ([], 0)
    for A in range(0,64):
        for B in range(0,64):
            for C in range(0,64):
                for D in range(0,64):
                    grade = fitness([A,B,C,D])
                    if grade > highest_grade[1]:
                        highest_grade = ([A,B,C,D], grade)
    
    return highest_grade
                    

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
