import matplotlib.pyplot as plt
%matplotlib qt
import numpy as np
import pandas as pd
import seaborn as sns
plt.style.use('dark_background')

##### CODE TO RUN: #####
## data_filepath = r'C:\Users\Maroon\Desktop\semester 5\236756 intro to ML\hw4\protein.csv'
## hw4_label_protein_data(data_filepath)
########################

def load_data(filepath):
    df = pd.read_csv(filepath, header=0)
    return df
    
def remove_outliers(df):
    Q1 = df.quantile(0.20)
    Q3 = df.quantile(0.80)
    IQR = Q3 - Q1
    trmask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    df[trmask] = np.nan
    return df
    
def imputate_data(df):
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=10)
    df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)
    return df
def graph_clusters(x,y,z, label):
    fig = plt.figure(figsize=(15,15))
    fig.suptitle('4,5,6 KMEANS', fontsize=20)
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, marker="X", c=label, cmap='coolwarm')
    
    plt.show()
    return
    
def get_clusters_kmeans(df):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=0)
    labels = kmeans.fit_predict(df)
    
    graph_clusters(df['protein_4'],df['protein_5'],df['protein_6'], labels)
    return labels, kmeans.cluster_centers_
    
def get_clusters_spectral(df):
    from sklearn.cluster import SpectralClustering
    clustering = SpectralClustering(n_clusters=5,
         assign_labels="discretize",
         random_state=0)
    return clustering.fit_predict(df)

def find_nearest_mutation(centroids, to):
    distance = -1
    closest = -1
    for i in range(0,len(centroids)):
        if i == to:
            continue
        distance_temp = 0
        for j in range(0, len(centroids[i])):
            distance_temp += abs(centroids[to][j]**2 - centroids[i][j])
            
        if(distance < 0 or distance_temp < distance):
            distance = distance_temp
            closest = i
    
    return closest, distance ** 0.5
        
def get_mutation_info(label, centroids):
    unique, counts = np.unique(label, return_counts=True)
    for i in range(0,len(centroids)):
        print('mutation ', i, ' info:')
        print('prevalence: ', counts[i] / len(label))
        print('centroid: ', centroids[i])
        c, d = find_nearest_mutation(centroids, i)
        print('nearest mutation: ', c, ' at distance: ', d, '\n')
    
def export_clusters(ids, labels):
    df = pd.concat([ids, pd.DataFrame(data=labels, columns=['y'])], axis=1)
    df.to_csv('clusters.csv', index=False)
    return

def export_top5_features(file):
    f = open(file, "w")
    f.write("'protein_4'\n'protein_5'\n'protein_6'\n'protein_8'\n'protein_9'")
    f.close()
    return
    
def hw4_label_protein_data(proteins_filepath):
    df = load_data(proteins_filepath)
    ids = df['ID'].copy(deep=True)
    data = remove_outliers(df.drop(columns=['ID']))
    data = imputate_data(data)
    kmeans_label, kmeans_centroids = get_clusters_kmeans(data)
    get_mutation_info(kmeans_label, kmeans_centroids)
    export_clusters(ids, kmeans_label)
    export_top5_features('selected_proteins.txt')
    return