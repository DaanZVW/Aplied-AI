 # Import library's
import numpy as np
import copy
from collections import Counter

# Er wordt geen random gebruikt dus wordt deze ook niet geïmporteerd.
# En daarbij dan ook niet geset op 0.

class Data_point:
    """This is an data class which is used for the AI. This saves the data columns and season, if provided.
    """

    def __init__( self,data,season='None' ):
        """Constructor for a Data_point, this sets the data and season variable

        Args:
            data (list): list of variables
            season (str, optional): Season where this data belongs to. Defaults to 'None'.
        """
        self.data = data
        self.season = season

class SeasonAI:
    """This class is an AI used to determine the season of the given dataset.
    """

    def __init__( self, seasons ):
        """
        Constructor for the AI, sets the seasons which are given
        """
        # Make the list for seasons
        self.seasons = seasons

        # Make copy of first row for column min and max list
        self.min_list = []
        self.max_list = []

        # Make list with all the data points for training data
        self.data_points = []


    def TrainAI( self, data, dates ):
        """Train the SeasonAI with a dataset

        Args:
            data (list): List with data for the AI
            dates (list): List with dates for the AI
        """
        # Make copy of first row for column min and max list
        self.min_list = copy.deepcopy(data[0])
        self.max_list = copy.deepcopy(data[0])

        # Go through all the unseen data rows and check if it is
        # higher or lower than the found min or max of that col
        for row in data[1:]:
            for colindex in range(len(row)):
                self.min_list[colindex] = min( row[colindex], self.min_list[colindex] )
                self.max_list[colindex] = max( row[colindex], self.max_list[colindex] )
        
        # Append data with data_points
        for row, date in zip(data, dates):
            normalised_data = []

            # Rescale all the data training data with min, max normalisation
            for col, min_value, max_value in zip(row, self.min_list, self.max_list):
                normalised_data.append( ((col - min_value) / (max_value - min_value)) )

            # Add a new data_point with dataobject_id, normalised data and the season which it is in
            self.data_points.append( Data_point( normalised_data, self.GetSeason(date) ) )


    def CheckDaySeason( self, data, size_k ):
        """Decider which season the data can find with the k-NN algorithm

        Args:
            data (list): List with data for the AI
            size_k (int): Size of neighbours group

        Returns:
            Str: Returns season which the algorithm found
        """
        normalised_data = []

        # Rescale all the data training data with min, max normalisation
        for col, min_value, max_value in zip(data, self.min_list, self.max_list):
            normalised_data.append( ((col - min_value) / (max_value - min_value)) )

        # Check all the data_points with the new data
        distance_data_point = []
        for data_point in self.data_points:

            totaldistance = 0
            for normalised_col, data_point_col in zip(normalised_data, data_point.data):
                totaldistance += (normalised_col - data_point_col)**2
            
            # Append the distance of the data_point to the total list, and include an
            # id so the when the list gets sorted we can trace back the data_point_id
            distance_data_point.append( [np.sqrt( totaldistance ), data_point.season] )        

        # Sort the distance_data_point on distance
        distance_data_point.sort(key=lambda x: x[0])

        # Get the size_k amount of neighbors
        nearest_neighbours = distance_data_point[:size_k]

        # Get all the seasons of all the other data_points 
        season_list = []
        for neighbour in nearest_neighbours:
            season_list.append( neighbour[1] )

        while True:

            # Count all the seasons found in season_list and get the 4 most common seasons
            season_counter = Counter(season_list)
            season_most_common = season_counter.most_common(4)

            # Check if the lenght of the most common season is 1, this means
            # that that season automaticly is chosen
            if len(season_most_common) == 1:

                # Return the most common season found in the nearest neighbours
                return season_most_common[0][0]

            # Check if the most common list has more than 1 highest number (most_common is
            # sorted so the next number can only be lower or the same)
            elif season_most_common[0][1] == season_most_common[1][1]:
                
                # Make the list 1 smaller for next cicle
                season_list = season_list[:-1]
                continue

            # Return the most common season found in the nearest neighbours
            return season_most_common[0][0]


    def GetSeason( self, date ):
        """Convert date to season

        Args:
            date (integer): Date integer, represents the time

        Returns:
            Str: Returns the season
        """
        # Convert the float date to int month
        date = int(str(int(date))[-4:])
        
        if date < 301:
            return self.seasons[0]
        elif 301 <= date < 601:
            return self.seasons[1]
        elif 601 <= date < 901:
            return self.seasons[2]
        elif 901 <= date < 1201:
            return self.seasons[3]
        else: # From 01-12 to end of year
            return self.seasons[0]


class SeasonAISupervisor:
    """Supervisor for the SeasonAI
    """

    def __init__( self, season_ai ):
        """Constructor for the supervisor

        Args:
            season_ai (Class): The season AI which is used
        """
        self.SeasonAI = season_ai

    def TestDaySeason( self, data, date, size_k ):
        """Test the day with the seasonAI and return if it was correct

        Args:
            data (list): List with data for the AI
            date (int): Date integer, represents the time
            size_k (int): Size of neighbours group

        Returns:
            (bool): Returns the correctness of the AI
        """
        found_season = self.SeasonAI.CheckDaySeason( data, size_k )
        actual_season = self.SeasonAI.GetSeason( date ) 
        return found_season == actual_season

    def TestYearSeason( self, data, dates, size_k ):
        """Test a whole dataset and return percentage correct

        Args:
            data (list): List of data for the AI 
            dates (list): List of dates for the AI
            size_k (int): Size of neighbours group

        Returns:
            (float): Percentage of correct found seasons
        """
        correct = 0

        for data_point, date in zip(data, dates):
            correct += self.TestDaySeason( data_point, date, size_k )
        
        return round(correct / len(data) * 100, 1)

    def TestBestKSize( self, data, dates, begin_size_k, end_size_k ):
        """Find the best k-size by returning begin_size_k to end_size_k percentages

        Args:
            data (list): List of data for the AI
            dates (list): List of data for the AI
            begin_size_k (int): Begin k_size where the function should start
            end_size_k (int): End k_size where the function should stop

        Returns:
            (list): List with all the percentages
        """
        # Return if begin_size_k is lower than 1
        if begin_size_k < 1 or end_size_k < 1:
            print( "Cant test k_size lower than 1!" )
            return [-1]

        # Loop through all the k-sizes
        test_results = []
        for size_k in range( begin_size_k, end_size_k ):
            test_results.append( self.TestYearSeason( data, dates, size_k ) )
        
        return test_results

    def GetPredictions( self, data, k_size ):
        """Get predictions of data without a date

        Args:
            data (list): List of data for the AI
            size_k (int): Size of neighbours group 

        Returns:
            (list): List with all the found seasons
        """
        results = []
        for data_point in data:
            results.append( self.SeasonAI.CheckDaySeason( data_point, k_size ) )
        
        return results


# Define our k_sizes which were gonna test
k_size_begin = 1
k_size_end = 365

# Get our dataset which we are gonna train our AI with
data_train = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
dates_train = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])

# Make our seasons
seasons = ['winter', 'lente', 'zomer', 'herfst']

# Make and train our AI
season_ai = SeasonAI( seasons )
season_ai.TrainAI( data_train, dates_train )

# Get our validation dataset which were gonna use to test our dataset
data_validation = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
dates_validation = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])

# Make a supervisor which is gonna test our AI
season_supervisor = SeasonAISupervisor( season_ai )
test_results = np.array(season_supervisor.TestBestKSize( data_validation, dates_validation, k_size_begin, k_size_end))

# Get the highest value of our test results
highest_test_result = max(test_results)

# Print our results in the terminal
print( test_results )
print( "Highest percentage found: {}".format( highest_test_result ) )
print( "With k-size(s):", end=" " )

for k_size_index in np.where( test_results == highest_test_result )[0]: 
    print( k_size_index + k_size_begin, end=" " )

# Get our validation dataset which were gonna use to test our dataset
data_prediction = np.genfromtxt('days.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

print( '\n', season_supervisor.GetPredictions( data_prediction, list(test_results).index( highest_test_result ) ) )
