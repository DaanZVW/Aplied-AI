 # Import library's
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
from kneed import KneeLocator


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


def import_dataset( filename: str, seasons: list ) -> list:
    """Determines the distance between 2 data points

    Args:
        filename (str): The name of the .csv file containing the data
        seasons (list): List of seasons

    Returns:
        data (list): List of data
        labels (list): List of labels
    """
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


def make_random_centroids( k: int, dataset: list ) -> list:
    """Determines the distance between 2 data points

        Args:
            k (int): amount of clusters
            dataset (list): List of data for the AI

        Returns:
            (list): List of centroids
        """
    return np.array(random.sample(list(dataset), k))


def get_distance( data_point_1: list, data_point_2: list ) -> float:
    """Determines the distance between 2 data points

    Args:
        data_point_1 (list): The first data point
        data_point_2 (list): The second data point

    Returns:
        total_distance (list): The total distance between 2 points
    """
    total_distance = 0
    for col_1, col_2 in zip( data_point_1, data_point_2 ):
        total_distance += (col_1 - col_2) ** 2
    return total_distance


def find_nearest_centroid( centroids: list, dataset: list, dates: list) -> list:
    """Finds the nearest centroid and adds the data point to its cluster

    Args:
        centroids (list): List of centroids
        dataset (list): List of data for the AI
        dates (list): List of dates for the AI

    Returns:
        clustered_data_points (list): List of new clusters
    """
    # Make nested list with amount of centroids
    clusterd_data_points = [[] for _ in range(len(centroids))]

    # Loop through all data_points
    for data_point, date in zip(dataset, dates):

        # Go through all centroids and put distance to centroid in list
        distance_to_centroids = []
        for centroid in centroids:
            distance_to_centroids.append( get_distance( data_point_1=data_point, data_point_2=centroid ) )

        # Find index of lowest distance and append data_point to that centroid
        data_point = Data_point(data_point, date)
        clusterd_data_points[ distance_to_centroids.index( min(distance_to_centroids) ) ].append( data_point )
    return clusterd_data_points


def get_new_centroids( clusters: list ) -> list:
    """Calculates new centroids

    Args:
        clusters (list): List of data clusters

    Returns:
        new_centroids (list): List of new centroids
    """
    new_centroids = []
    for cluster in clusters:
        average = [0.0 for index in range(7)]
        for i in range(0, len(cluster)):
            for j in range(0, len(cluster[i].data)):
                average[j] += cluster[i].data[j]
        average[:] = [x / len(cluster) for x in average]
        new_centroids.append(average)
    return np.array(new_centroids)


def k_means( k: int, dataset: list, dates: list) -> list:
    """Clusters the data in k clusters based on the season

    Args:
        k (int): amount of clusters
        dataset (list): List of data for the AI
        dates (list): List of dates for the AI

    Returns:
        cluster (list): List with a cluster
        centroids (list): List of centroids
    """
    centroids = make_random_centroids( k=k, dataset=dataset )
    while True:
        clusters = find_nearest_centroid(centroids, dataset, dates)
        new_centroids = get_new_centroids(clusters)
        if (centroids == new_centroids).all():
            break
        centroids = new_centroids
    return clusters, centroids


def cluster_into_seasons(clusters: list):
    """Use the maximum vote principle to cluster the data into the 4 different seasons

    Args:
        clusters (list): List of data for the AI_

    Returns:
        (list): List with a cluster and its season
    """
    season_of_clusters = []
    for cluster in clusters:
        season_of_items = []
        for item in cluster:
            season_of_items.append(item.season)
        most_common_season = [item for item in Counter(season_of_items).most_common()]
        season_of_clusters.append([most_common_season[0][0], cluster])
    return season_of_clusters

def calculate_season_accuracy( clusters: list ):
    """Calculate the accuracy's of the seasons

    Args:
        clusters (list): The calculated clusters created by the k-means algorithm
    """
    new_season_clusters = [[clusters[index][0],[]] for index in range(0, 4)]
    for season_index ,season_cluster in enumerate(clusters):
        for data_point_season in season_cluster[1]:
            new_season_clusters[season_index][1].append( data_point_season.season )

    for season_cluster in new_season_clusters:
        most_common_season = [season for season in Counter(season_cluster[1]).most_common()]

        total_found = 0
        most_common_found = most_common_season[0][1]
        for season in most_common_season:
            total_found += season[1]
    
        print( "Season {} had an accuracy of: {}".format( most_common_season[0][0], round((most_common_found / total_found)*100, 1)))


if __name__ == "__main__":
    random.seed( 42 )
    k_min = 2
    k_max = 50
    dataset, dates = import_dataset( filename="dataset1.csv", seasons=["winter","lente","zomer","herfst"])

    result, centroids = k_means(k=4, dataset=dataset, dates=dates)

    season_clusters = cluster_into_seasons(result)

    calculate_season_accuracy( season_clusters )

    diff = {}
    for k in range(k_min, k_max):
        cluster_dist = []

        clusters, centroids = k_means(k, dataset, dates)
        for cluster, centroid in zip(clusters, centroids):
            cluster_dist.append(sum([get_distance(point.data, centroid) ** 2 for point in cluster]))
        diff[k] = np.mean(cluster_dist)

    lists = sorted(diff.items())
    x, y = zip(*lists)
    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    print(kn.knee)
    plt.xlabel('k')
    plt.ylabel('value')
    plt.plot(x, y)
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()