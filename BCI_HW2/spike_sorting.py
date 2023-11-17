import numpy as np
from matplotlib import pyplot as plt
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from get_ori_data import gen_wf_data
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,recall_score,f1_score,classification_report

def plot_wf(wf,y):
    x=np.arange(64)
    plt.plot(x,wf)
    plt.show()

def k_means(X_train, X_test, y_train, y_test,x,x_,y):
    color=['green','blue','orange','red']
    n_clusters = 4
    cluster = KMeans(n_clusters=n_clusters, random_state=0)

    y_pred=cluster.fit_predict(x)
    plt.scatter(x_[:, 0], x_[:, 1], c=y_pred)
    plt.show()
    evalue(y_pred,y)


    y_pred=cluster.fit_predict(x_)
    plt.scatter(x_[:, 0], x_[:, 1], c=y_pred)
    plt.show()
    evalue(y_pred,y)

def plot_PCA(x,y):
    pca = decomposition.PCA(n_components=2)  # n_components：目标维度，需要降维成n_components个特征
    x_ = pca.fit_transform(x)  # 生成降维后的新数据

    color=['green','blue','orange','red']
    print(x_.shape)
    plt.scatter(x_[:,0],x[:,1],c=(np.vstack(y))-1)
    plt.show()
    return x_

def evalue(label_,label):
    # 代码 6-12
    from sklearn.metrics import fowlkes_mallows_score

    score = fowlkes_mallows_score(np.squeeze(label), label_)
    print('数据聚类FMI评价分值为：%f' % (score))

    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    silhouettteScore = []

    score = silhouette_score(label, label_)
    print('数据聚类silhouettteScore评价分值为：%f' % (score))

    # 代码 6-14
    from sklearn.metrics import calinski_harabaz_score

    score = calinski_harabaz_score(np.squeeze(label), label_)
    print('iris数据聚calinski_harabaz指数为：%f' % (score))


if __name__ == '__main__':
    data_path='Zenodo/indy_20161024.mat'
    x,y=gen_wf_data(data_path)
    X_train, X_test, y_train, y_test=train_test_split(x,y,train_size=0.75)

    x_=plot_PCA(x,y)
    # x_=x
    k_means(X_train, X_test, y_train, y_test,x,x_,y)