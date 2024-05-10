import numpy as np

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization

    chosen_rows = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[chosen_rows]
    return centroids

def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization

    centroids = []
    # Pierwsza centroida wybierana losowo
    centroids.append(data[np.random.randint(data.shape[0])])
    
    for _ in range(1, k):
        average_distances = []
        for point in data:
            distances_to_centroids = [np.linalg.norm(point - centroid) for centroid in centroids]
            # Srednia odleglosc do wszystkich pozostalych centroidow
            avg_distance = np.mean(distances_to_centroids)
            average_distances.append(avg_distance)
        
        max_avg_distance_index = np.argmax(average_distances)
        new_centroid = data[max_avg_distance_index]
        centroids.append(new_centroid)
    
    return np.array(centroids)

def assign_to_cluster(data, centroids):
    # TODO find the closest cluster for each data point
    
    assignments = np.zeros(len(data), dtype=int)
    
    for i, point in enumerate(data):
        # Norma euklidesowa - pierwiastek sumy kwadratow funkcja norm()
        distances = np.linalg.norm(centroids - point, axis=1)
        
        # Sprawdzenie minimalnej wartosci w distances
        cluster_index = np.argmin(distances)
        assignments[i] = cluster_index
    
    return assignments

def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments

    new_centroids = []
    
    for cluster_index in np.unique(assignments):

        cluster_points = data[assignments == cluster_index]
        cluster_mean = np.mean(cluster_points, axis=0)
        # Nowy centroid to srednia z punktow, do ktorych byl przypisany
        new_centroids.append(cluster_mean)
    
    return np.array(new_centroids)

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        #print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

