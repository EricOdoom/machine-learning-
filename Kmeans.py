from random import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
inf = float("inf")

class ClusterPoint:
    def __init__(self, coords, marked=False):
        self.coords = coords
        self.cluster = None
        self.marked = marked

    def __repr__(self):
        return "<Coords: "+str(self.coords)+", Cluster: "+str(self.cluster)+">"

    def __str__(self):
        return "<Coords: "+str(self.coords)+", Cluster: "+str(self.cluster)+">"

def calc_all_centroids(points, k):
    '''
        Returns length k list, where list[k] = centroid of kth cluster
    '''

    dim = len(points[0].coords) #get dimension of points

    cluster_centroids = []
    for i in range(k):
        cluster_centroids.append([0]*dim)

    cluster_point_count = [0]*k

    for point in points:
        if point.cluster is not None:
            cluster_point_count[point.cluster] += 1
            for coord_index in range(dim):
                cluster_centroids[point.cluster][coord_index] += point.coords[coord_index]

    #divides the total sum of points in each cluster by the number of points_temp
    #in that cluster
    for i in range(k):
        for j in range(dim):
            if cluster_point_count[i] > 0:
                cluster_centroids[i][j] = cluster_centroids[i][j] / cluster_point_count[i]
            else:
                print("EMPTY CLUSTER ALERT!")
                cluster_centroids[i][j] = None

    return cluster_centroids

def dist_between(p1, p2):
    total_sum = 0
    for i in range(len(p1)):
        total_sum += abs(p1[i]-p2[i])
    return total_sum**(.5)

def show_plot(x_vals, y_vals, colors, markers, fig):
    plt.title("K-Means Clustering")
    plt.scatter(x_vals, y_vals, c=colors, s=markers)
    plt.show()


all_graphs = []

def calc_graph(points, k, colors_array):
    scatter_x = []
    scatter_y = []
    colors = []
    markers = []

    for point in points:
        scatter_x.append(point.coords[0])
        scatter_y.append(point.coords[1])
        colors.append(colors_array[point.cluster])
        if point.marked:
            markers.append(40)
        else:
            markers.append(20)

    return [scatter_x, scatter_y, colors, markers]

cur_click_count = 0
def click_event(event):
    global cur_click_count
    event.canvas.figure.clear()
    event.canvas.figure.gca().scatter(all_graphs[cur_click_count][0], all_graphs[cur_click_count][1],
    c=all_graphs[cur_click_count][2], s=all_graphs[cur_click_count][3])
    cur_click_count+=1
    if (cur_click_count>=len(all_graphs)):
        cur_click_count = 0
    event.canvas.draw()




def k_means_simple(points, k):
    global all_graphs
    all_graphs = []
    #choose centroids: currently, does in order (should change)
    for i in range(k):
        points[i].cluster = i

    cluster_centroids = calc_all_centroids(points, k)

    assignments_changed = True
    while(assignments_changed):
        assignments_changed = False

        #   assignment step:
        #       iterate over the points, assigning each to their best
        #       cluster
        for point in points:
            min_dist = inf
            best_cluster = -1
            for j in range(k):
                cur_dist = dist_between(point.coords, cluster_centroids[j])
                if (cur_dist < min_dist):
                    min_dist = cur_dist
                    best_cluster = j
            if point.cluster != best_cluster:
                point.cluster = best_cluster
                assignments_changed = True
        #   update step:
        #       recalculate cluster centroids
        all_graphs.append(calc_graph(points, k, ["red", "green", "blue"]))
        cluster_centroids = calc_all_centroids(points, k)

    # all_graphs.pop()

    dim = len(points[0].coords)
    if (dim == 2):
        colors_array = ["red", "blue", "green"] #make auto colors up to k later
        scatter_x, scatter_y, colors, markers = calc_graph(points, k, colors_array)
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.mpl_connect('button_press_event', click_event)
        show_plot(scatter_x, scatter_y, colors, markers, fig)

    if False:
        for point in points:
            scatter_x.append(point.coords[0])
            scatter_y.append(point.coords[1])
        y_pred = KMeans(n_clusters=2).fit_predict(points.coords)
        plt.scatter(scatter_x, scatter_y, c=y_pred)
        plt.show()

    if (dim == 3):
        colors_array = ["red", "blue", "green"] #make auto colors up to k later
        scatter_x = []
        scatter_y = []
        scatter_z = []
        colors = []
        markers = []
        for point in points:
            scatter_x.append(point.coords[0])
            scatter_y.append(point.coords[1])
            scatter_z.append(point.coords[2])
            colors.append(colors_array[point.cluster])
            if point.marked:
                markers.append(40)
            else:
                markers.append(20)
        fig = plt.figure()
        ax = plt.axes(projection="3d", xlabel = 'texture', ylabel='mean radius', zlabel='mean perimeter')
        ax.scatter3D(scatter_x, scatter_y, scatter_z, s=markers, c=colors)
        plt.show()

def k_means_skl(points, k):
    scatter_2d = []
    for point in points:
        scatter_2d.append([point.coords[0], point.coords[1]])
    scatter_2d=np.array(scatter_2d)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scatter_2d)
    plt.scatter(scatter_2d[:,0], scatter_2d[:,1], c=kmeans.labels_, cmap='rainbow')
    plt.show()

def mini_batch_k_means(points, k):
    scatter_2d = []
    for point in points:
        scatter_2d.append([point.coords[0], point.coords[1]])
    scatter_2d = np.array(scatter_2d)
    mbkm = MiniBatchKMeans(n_clusters=k)  # Take a good look at the docstring and set options here
    mbkm.fit(scatter_2d)
    plt.scatter(scatter_2d[:, 0], scatter_2d[:, 1], c=mbkm.labels_, cmap='rainbow')
    plt.show()

def agglo(points, k):
    scatter_2d = []
    for point in points:
        scatter_2d.append([point.coords[0], point.coords[1]])
    scatter_2d = np.array(scatter_2d)
    agg = AgglomerativeClustering(n_clusters=k)
    agg.fit(scatter_2d)
    plt.scatter(scatter_2d[:, 0], scatter_2d[:, 1], c=agg.labels_, cmap='rainbow')
    plt.show()

        #Things to try (Would be cool!):
        # - 3D!
        # - the LINES?
        # - slideshow/gif!
        # - better initial centroid choices?
        # - auto k?
        # - compare to built-in scikitlearn.KMeans
        # - test on real data


def process_data(filename):
    '''Replaced malignant with 1 and benign with 0.'''
    dataset = pd.read_csv(filename)
    dataset = dataset.fillna(dataset.mean())
    map = {'B':0, 'M':1}
    dataset = dataset.applymap(lambda s: map.get(s) if s in map else s)
    texture = dataset['texture_worst']
    radius_mean = dataset['perimeter_se']
    perimeter = dataset['perimeter_mean']
    diagnosis = dataset['diagnosis']
    return texture, radius_mean, perimeter, diagnosis

def breast_cancer():
    texture, radius_mean, perimeter, diagnosis = process_data("data.csv")
    coord_list = [(x,y,z,p) for x,y,z,p in zip(texture,radius_mean, perimeter,diagnosis)]
    points = []

    for c in coord_list:
        if c[3] == 1:
            points.append(ClusterPoint(c[:3], marked=True))
        else:
            points.append(ClusterPoint(c[:3], marked=False))

    k_means_simple(points,2)

def test_random_data(points = 100, grid_size = 50, k=3, show_their_plot=True, show_mini_batch=True, show_agglomerative_clustering = True):
    coord_list = [(randrange(grid_size), randrange(grid_size)) for i in range(points)]

    points = []
    for coord in coord_list:
        points.append(ClusterPoint(coord))

    k_means_simple(points,3)

    if (show_their_plot):
        k_means_skl(points, 3)
    if (show_mini_batch):
        mini_batch_k_means(points, 3)
    if (show_agglomerative_clustering):
        agglo(points, 3)

def test_special_data():
    coord_list = []
    for i in range(8):
        coord_list.append((randrange(10,15), randrange(10,15)))
    for i in range(8):
        coord_list.append((randrange(0,5), randrange(8,11)))
    for i in range(8):
        coord_list.append((randrange(10,15), randrange(0,4)))

    points = []
    for coord in coord_list:
        points.append(ClusterPoint(coord))

    k_means_simple(points,3, marked=True)

def test_special_data_3d():
    coord_list = []
    for i in range(8):
        coord_list.append((randrange(10,15), randrange(10,15), randrange(10,15)))
    for i in range(8):
        coord_list.append((randrange(0,5), randrange(8,11), randrange(8,11)))
    for i in range(8):
        coord_list.append((randrange(10,15), randrange(0,4), randrange(0,4)))

    points = []
    for coord in coord_list:
        points.append(ClusterPoint(coord))

    k_means_simple(points,3)

def test_markers_3d():
    coord_list = [(randrange(10), randrange(10), randrange(10)) for i in range(20)]

    points = []
    for coord in coord_list[0:10]:
        points.append(ClusterPoint(coord, marked=True))
    for coord in coord_list[10:20]:
        points.append(ClusterPoint(coord, marked=False))

    k_means_simple(points,3)


def main():
    # test_markers_3d()
    breast_cancer()
    # test_special_data()
    # test_special_data_3d()
    test_random_data()
    # test_random_data(10, 5, 3)



if __name__ == "__main__":
    main()
