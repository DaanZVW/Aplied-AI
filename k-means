 # Import library's
import numpy as np
import copy
import random

def import_dataset( filename: str, seasons: list ) -> list:
    data = np.genfromtxt(
        filename, 
        delimiter=';', 
        usecols=[1,2,3,4,5,6,7], 
        converters={
            5: lambda s: 0 if s == b"-1" else float(s), 
            7: lambda s: 0 if s == b"-1" else float(s)
        })
    
    dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
   
    labels = []
    for date in dates:

        # Convert the float date to int month
        date = int(str(int(date))[-4:])

        if date < 301:
            labels.append(seasons[0])
        elif 301 <= date < 601:
            labels.append(seasons[1])
        elif 601 <= date < 901:
            labels.append(seasons[2])
        elif 901 <= date < 1201:
            labels.append(seasons[3])
        else: # From 01-12 to end of year
            labels.append(seasons[0])

    return data, labels

def get_min_max_dataset( dataset: list ) -> list:
    # Make copy of first row for column min and max list
    min_list = copy.deepcopy(dataset[0])
    max_list = copy.deepcopy(dataset[0])

    # Go through all the unseen data rows and check if it is
    # higher or lower than the found min or max of that col
    for row in dataset[1:]:
        for colindex in range(len(row)):
            min_list[colindex] = min( row[colindex], min_list[colindex] )
            max_list[colindex] = max( row[colindex], max_list[colindex] )
        
    return min_list, max_list

def make_random_centroids( k: int, dataset: list ) -> list:
    centroids = []

    min_list, max_list = get_min_max_dataset( dataset=dataset )

    for _ in range( k ):
        temp_centroid = []
        for col_min, col_max in zip( min_list, max_list ):
            temp_centroid.append( random.randint( col_min, col_max ) )
        centroids.append( temp_centroid )
        
    return np.array(centroids)

def get_distance( data_point_1: list, data_point_2: list ) -> float:
    total_distance = 0
    for col_1, col_2 in zip( data_point_1, data_point_2 ):
        total_distance += (col_1 - col_2) ** 2
    return total_distance

def find_nearest_centroid( centroids: list, dataset: list ) -> list:
    # Make nested list with amount of centroids
    clusterd_data_points = [[] for _ in range(len(centroids))]

    # Loop through all data_points
    for data_point in dataset:

        # Go through all centroids and put distance to centroid in list
        distance_to_centroids = []
        for centroid in centroids:
            distance_to_centroids.append( get_distance( data_point_1=data_point, data_point_2=centroid ) )
        
        # Find index of lowest distance and append data_point to that centroid
        clusterd_data_points[ distance_to_centroids.index( min(distance_to_centroids) ) ].append( data_point )

    return clusterd_data_points

def get_new_centroids( clusters: list ) -> list:
    new_centroids = []
    for cluster in clusters:
        if len(cluster) == 0:
            

def k_means( k: int, dataset: list ) -> list:
    centroids = make_random_centroids( k=k, dataset=dataset )
    # new_centroids = np.array()

    # while (centroids != new_centroids).all():
    clusters = find_nearest_centroid( centroids, dataset )
    print(clusters)


if __name__ == "__main__":
    random.seed( 42 )

    dataset, dates = import_dataset( filename="dataset1.csv", seasons=["winter","lente","zomer","herfst"])
    print ( k_means( k=2, dataset=dataset ) )


