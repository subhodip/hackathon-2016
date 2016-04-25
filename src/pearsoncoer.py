import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import scipy.sparse as sps
from sklearn.cross_validation import   train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

class pcr:

    def get_mse(self, pred, actual):
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual)

    def sparsity(self):
        data = pd.read_csv('data-files/raw_data_march.csv')
        data.drop(data.columns[[0, 1, 2, 3, 4, 7, 12, 13, 14, 15, 16, 17, 18, 19]], axis=1, inplace=True)
        #df = data.pivot_table(index=['Practitioner'], columns=['ISBN'], values=['Qty'])
        #print (df)
        user = list(data['Practitioner'].unique())
        isbn = list(data['ISBN'].unique())

        d = data['Qty'].tolist()
        row = data.Practitioner.astype('category', categories=user).cat.codes
        col = data.ISBN.astype('category', categories=isbn).cat.codes
        sparse_matrix = sps.csr_matrix((d, (row, col)), shape=(len(user), len(isbn)))

        dfs=pd.SparseDataFrame([ pd.SparseSeries(sparse_matrix[i].toarray().ravel(), fill_value=0)
                              for i in np.arange(sparse_matrix.shape[0]) ], index=user, columns=isbn, default_fill_value=0)

        #print (dfs)
        # calculate sparsity here
        print (dfs.density)
        # calculation ends
        return dfs, data

    def train_test_split(self, dfs):
        train, test = train_test_split(dfs, test_size = 0.2)
        return train, test

    def fast_sim(self, train, kind='user', epsilon=1e-9):
        dfs = train.as_matrix()
        if kind == 'user':
            sim = dfs.dot(dfs.T) + epsilon
        elif kind == 'item':
            sim = dfs.T.dot(dfs) + epsilon
        norms = np.array([np.sqrt(np.diagonal(sim))])
        return  (sim / norms / norms.T)

    def predict_fast_simple(self, dfs, simmilarity, kind='user'):
        data = dfs.as_matrix()
        if kind == 'user':
            return simmilarity.dot(data) / np.array([np.abs(simmilarity).sum(axis=1)]).T
        elif kind == 'item':
            return data.dot(simmilarity) / np.array([np.abs(simmilarity).sum(axis=1)])

    def predict_topk(self, dfs, similarity, kind='user', k=11):
        #print (dfs)
        ratings = dfs.as_matrix()
        pred = np.zeros(ratings.shape)
        if kind == 'user':
            for i in range(ratings.shape[0]):
                top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
                for j in range(ratings.shape[1]):
                    pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                    pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))

        if kind == 'item':
            for j in range(ratings.shape[1]):
                top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
                for i in range(ratings.shape[0]):
                    pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                    pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))

        return pred

    def top_k_users(self, similarity, map, idx, k=10):
        return [map.dtypes.index[x] for x in np.argsort(similarity[idx,:])[:-k-1:-1]]




def main():
    k_array = [5, 15, 30, 50, 100, 200]
    user_train_mse = []
    user_test_mse = []
    item_test_mse = []
    item_train_mse = []
    p = pcr()
    dfs, orig_data = p.sparsity()
    train, test  = p.train_test_split(dfs)
    item_sim = p.fast_sim(train, kind='item')
    user_sim = p.fast_sim(train, kind='user')
    user_predict = p.predict_fast_simple(train, user_sim, kind='user')
    u_pred_topk = p.predict_topk(train, user_sim, kind='user', k=40)
    i_pred_topk = p.predict_topk(train, item_sim, kind='item', k=40)
    print ('Top-k User-based CF MSE: ' + str(p.get_mse(u_pred_topk, test.as_matrix())))
    print ('Top-k Item-based CF MSE: ' + str(p.get_mse(i_pred_topk, test.as_matrix())))
    #print (u_pred_topk)
    l = p.top_k_users(item_sim, dfs, idx=10)
    #u = p.top_k_users(user_sim, dfs, idx=0)
    print (l)
    #print (u)
    for isbn in l:
        print (orig_data.loc[orig_data['ISBN'] == isbn]['Subtest'].unique())

    for k in k_array:
        user_pred = p.predict_topk(train, user_sim, kind='user', k=k)
        item_pred = p.predict_topk(train, item_sim, kind='item', k=k)

        user_train_mse += [p.get_mse(user_pred, train.as_matrix())]
        user_test_mse += [p.get_mse(user_pred, test.as_matrix())]

        item_train_mse += [p.get_mse(item_pred, train.as_matrix())]
        item_test_mse += [p.get_mse(item_pred, test.as_matrix())]


    sns.set()

    pal = sns.color_palette("Set2", 2)

    plt.figure(figsize=(8, 8))
    plt.plot(k_array, user_train_mse, c=pal[0], label='User-based train', alpha=0.5, linewidth=5)
    plt.plot(k_array, user_test_mse, c=pal[0], label='User-based test', linewidth=5)
    plt.plot(k_array, item_train_mse, c=pal[1], label='Item-based train', alpha=0.5, linewidth=5)
    plt.plot(k_array, item_test_mse, c=pal[1], label='Item-based test', linewidth=5)
    plt.legend(loc='best', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('k', fontsize=30)
    plt.ylabel('MSE', fontsize=30)
    plt.show()


if __name__ == '__main__':
    main()