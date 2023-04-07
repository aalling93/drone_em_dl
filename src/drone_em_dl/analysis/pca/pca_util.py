from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt 


def plot_explained_var(data,explained_var:float=0.95):
    pca = PCA().fit(data[:,:])
    plt.figure(figsize=(12,9))
    #plt.subplots()
    xi = np.arange(0, data[:,:].shape[1], step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='k')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, data[:,:].shape[1], step=1)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (\%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=explained_var, color='r', linestyle='-')
    pct = int(explained_var*100)
    plt.text(0.5, y[np.where(y>explained_var)[0][0]]+0.05, f'{pct} \% cut-off threshold', color = 'red')
    ax = plt.gca()
    ax.grid(axis='x')
    plt.show()




def plot_pcs(pca_data,pca_amount,explained_variance_ratio_):

    labels = [f'PC {i+1} {var}' for i, var in enumerate(explained_variance_ratio_ * 100)]
    fig = px.scatter_matrix(
        pca_data,
        color=pca_data[:,-1],
        dimensions=range(pca_amount),
        labels=labels,
        color_continuous_scale='jet' , #[(0,'#030F4F'), (0.5, '#DADADA'), (1,'#990000')]
        title=f'somehting',
        
    )
    fig.update_traces(diagonal_visible=True,showupperhalf = True)

    fig.update_layout(
        title='PC components',
        width=1600,
        height=1600,
    )

    fig.show()