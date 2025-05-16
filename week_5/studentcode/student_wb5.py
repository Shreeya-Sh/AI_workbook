# DO NOT change anything except within the function
from approvedimports import *

def cluster_and_visualise(datafile_name:str, K:int, feature_names:list):
    """Function to get the data from a file, perform K-means clustering and produce a visualisation of results.

    Parameters
        ----------
        datafile_name: str
            path to data file

        K: int
            number of clusters to use
        
        feature_names: list
            list of feature names

        Returns
        ---------
        fig: matplotlib.figure.Figure
            the figure object for the plot
        
        axs: matplotlib.axes.Axes
            the axes object for the plot
    """
   # ====> insert your code below here

    data = np.genfromtxt(datafile_name, delimiter=",")

    # creates a K-Means model with K clusters
    kmeans = KMeans(n_clusters=K, random_state=0)

    # fit the model to the data and get cluster labels
    cluster_ids = kmeans.fit_predict(data)
    
    num_feat = data.shape[1]

    # create figure and grid of plots
    fig, ax = plt.subplots(num_feat, num_feat, figsize=(12, 12))

    plt.set_cmap('viridis')
    hist_col = plt.get_cmap('viridis', K).colors

    for feature1 in range(num_feat):
        # y-axis for first col and x-axis for first row
        ax[feature1, 0].set_ylabel(feature_names[feature1])
        ax[0, feature1].set_xlabel(feature_names[feature1])
        ax[0, feature1].xaxis.set_label_position('top') # x label on top

        for feature2 in range(num_feat):
            x_data = data[:, feature1]
            y_data = data[:, feature2]

            # makes a scatter plot if the features are not same
            if feature1 != feature2:
                ax[feature1, feature2].scatter(x_data, y_data, c=cluster_ids, s=10)
            
            # makes histogram, if features are same
            else:
                # arranges the same clusters to stay together
                inds = np.argsort(cluster_ids)
                sorted_clusters = cluster_ids[inds]
                x_data = x_data[inds]

                splits = np.split(x_data, np.unique(sorted_clusters, return_index=True)[1][1:])
                
                # different color for different clusters
                for i, split in enumerate(splits):
                    ax[feature1, feature2].hist(split, bins=20, color=hist_col[i], edgecolor='black')


    # saves it to file 
    fig.suptitle(f"Visualisation of {K} clusters by s9-shakya", fontsize=16, y=0.925)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("myVisualisation.jpg")

    
    return fig,ax
    
    # <==== insert your code above here


