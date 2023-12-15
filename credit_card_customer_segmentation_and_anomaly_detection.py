# first step is to import the necessary modules
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

# define functions to plot the categorical columns and the numerical columns
def plot_categorical_columns(df,list_of_columns):
    for column_name in categorical_columns:
        if df[column_name].nunique() < 10:
            sns.countplot(df[column_name])
            plt.show()
        else:
            print(f'{column_name} has {df[column_name].nunique()} unique values')
            print(df[column_name].value_counts(normalize=True))

def plot_numerical_columns(df,list_of_columns):
    for column_name in integer_float_columns:
        sns.histplot(df[column_name])
        plt.show()
        sns.boxplot(df[column_name])
        plt.show()

def print_data_info(df):
    print(df.shape)
    print(df.columns)
    print(df.head())
    print(df.info())
    print(df.isnull().sum())


def hopkins_statistic(X):
    n, d = X.shape
    m = int(0.1 * n)  # using 10% of data for the test, can be adjusted

    # Generate random points
    np.random.seed(0)
    random_points = np.random.uniform(size=(m, d))

    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    
    # Find distance for real data points
    distances_real, _ = nbrs.kneighbors(X, n_neighbors=2)
    sum_distances_real = np.sum(distances_real[:, 1])

    # Find distance for random points
    distances_random, _ = nbrs.kneighbors(random_points, n_neighbors=1)
    sum_distances_random = np.sum(distances_random)

    # Calculate Hopkins statistic
    hopkins_stat = sum_distances_random / (sum_distances_real + sum_distances_random)

    return hopkins_stat

def plot_correlation(df):
    plt.figure(figsize=(20,20))
    sns.heatmap(df.corr(),annot=True)
    plt.show()


def find_optimal_clusters(df_pca):
    """
    Find the optimal number of clusters using the Elbow method and Silhouette method.

    Parameters:
    df_pca (DataFrame): The data for which you want to find the optimal number of clusters.
    """
    wcss = []  # List to store the Within-Cluster Sum of Squares
    silhouette_scores = []  # List to store Silhouette scores

    # Loop through a range of cluster numbers from 2 to 10
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(df_pca)
        wcss.append(kmeans.inertia_)  # Append WCSS to the list
        silhouette_scores.append(silhouette_score(df_pca, kmeans.labels_))  # Append Silhouette score to the list

    # Plot the Elbow Method graph
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, 11), wcss, marker='o', linestyle='-', color='b')
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')

    # Plot the Silhouette Method graph
    plt.subplot(1, 2, 2)
    plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-', color='g')
    plt.title('The Silhouette Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()

def apply_pca(df,n_components):
    pca = PCA(n_components=n_components)
    pca.fit(df)
    df_pca = pd.DataFrame(pca.transform(df),columns=['0','1'])
    return df_pca

def scale_data(df):
    scaler=StandardScaler()
    x_scaled = scaler.fit_transform(df)
    # get the scaled data into a dataframe
    df_scaled = pd.DataFrame(x_scaled,columns=df.columns)
    return df_scaled

def train_kmeans(df_pca,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters,random_state=42)
    kmeans.fit(df_pca)
    return kmeans

def find_optimal_eps(df):
    nbrs = NearestNeighbors(n_neighbors=2).fit(df)
    distances, indices = nbrs.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()

# 2. read and review the data
df = pd.read_csv(r'C:\Users\nirro\Downloads\Credit Card Dataset for Clustering\CC GENERAL.csv',index_col=False)

print_data_info(df)
# 2.1 drop the columns that are not needed
df.drop(['CUST_ID'],axis=1,inplace=True)
print(df.columns)

# 3. EDA - exploratory data analysis
list_of_columns = df.columns.tolist()

# 3.2 plot the categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
plot_categorical_columns(df,categorical_columns)

# 3.3 plot the numerical columns has pairs of histogram and boxplot
integer_float_columns = df.select_dtypes(include=['int64','float64']).columns.tolist()
plot_numerical_columns(df,integer_float_columns)


# Since the dataset is about clustering, imputation will use KNNImputer() to avoid biased clustering results. 
# The mean value from the nearest n_neighbors found in the dataset is used to impute the missing values for each sample.
# 3.4 fill missing values with KNNImputer
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df),columns=df.columns)
print(df.isnull().sum())

# preforme the hopkins test to check if the data is suitable for clusrering
if hopkins_statistic(df) < 0.5:
    print(f'hopkins value is {hopkins_statistic(df)}, The data is suitable for clustering')
else:
    print(f'hopkins value is {hopkins_statistic(df)},The data is not suitable for clustering')
    

# 3.5 find correlations between the columns using heatmap
plot_correlation(df)

# 4 pre-processing the data befor PCA and clustering
# 4.1 scale the data
df_scaled = scale_data(df)
print(df_scaled.head())

# 4.2 apply PCA 
df_pca = apply_pca(df_scaled,2)
print(df_pca.head())

# 5. clustering the data
# 5.2 find the optimal number of clusters using elbow method
find_optimal_clusters(df_pca)

# 5.3 train the model with the optimal number of clusters,add the cluster labels to the pca dataframe
df_pca['cluster'] = train_kmeans(df_pca,4).labels_
print(df_pca.head())
# 5.5 plot the clusters
sns.scatterplot(x='0',y='1',data=df_pca,hue=df_pca['cluster'])
plt.show()

# use different clustering algorithms for and do part 5.3-5.5 for each one
# 5.3 train the model with the optimal number of clusters
# find the optimal eps and min_samples for dbscan
find_optimal_eps(df_pca)

dbscan = DBSCAN(eps=2,min_samples=4)
dbscan.fit(df_pca)
# 5.4 add the cluster labels to the pca dataframe
df_pca['cluster_dbscan'] = dbscan.labels_
# 5.5 plot the clusters
sns.scatterplot(x='0',y='1',data=df_pca,hue=df_pca['cluster_dbscan'])
plt.show()

# 5.3 train the model with the optimal number of clusters
agglomerative = AgglomerativeClustering(n_clusters=3)
agglomerative.fit(df_pca)
# 5.4 add the cluster labels to the pca dataframe
df_pca['cluster_agglomerative'] = agglomerative.labels_
# 5.5 plot the clusters
sns.scatterplot(x='0',y='1',data=df_pca,hue=df_pca['cluster_agglomerative'])
plt.show()

# evaluate teh models performence using silhouette score,bouldin score and calinski harabasz score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

def evaluate_model(df,list_cluster_column):
    list_result = []
    for cluster_column in list_cluster_column:
        # print(f'evaluate model {cluster_column}')
        # print(f'silhouette score is {silhouette_score(df,df[cluster_column])}')
        # print(f'calinski harabasz score is {calinski_harabasz_score(df,df[cluster_column])}')
        # print(f'davies bouldin score is {davies_bouldin_score(df,df[cluster_column])}\n')
        list_result.append([cluster_column,silhouette_score(df,df[cluster_column]),calinski_harabasz_score(df,df[cluster_column]),davies_bouldin_score(df,df[cluster_column])])
    # add the results to a dataframe
    df_result = pd.DataFrame(list_result,columns=['cluster_column','silhouette_score','calinski_harabasz_score','davies_bouldin_score'])
    return df_result
print(evaluate_model(df_pca,['cluster','cluster_dbscan','cluster_agglomerative']).head())


# The next step is to evaluate the clustering quality provided by K-Means. Quality evaluation will use the Davies-Bouldin index, silhouette score, and Calinski-Harabasz index.
# Davis-Bouldin Index is a metric for evaluating clustering algorithms. 
# It is defined as a ratio between the cluster scatter and the cluster's separation. 
# Scores range from 0 and up. 0 indicates better clustering.

# Silhouette Coefficient/Score is a metric used to calculate the goodness of a clustering technique. 
# Its value ranges from -1 to 1. The higher the score, the better. 1 means clusters are well apart from each other and clearly distinguished. 
# 0 means clusters are indifferent/the distance between clusters is not significant. -1 means clusters are assigned in the wrong way.

# Calinski-Harabasz Index (also known as the Variance Ratio Criterion), 
# is the ratio of the sum of between-clusters dispersion and of inter-cluster dispersion for all clusters, 
# the higher the score, the better the performances.

# plot the results and add in the plot the optimal score for each metric

