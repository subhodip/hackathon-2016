#!/usr/bin/env python
# Read through csv data files and convert them to json, this is a slow process and it is recommended to used a solid
# database to churn out the data.
import csv
import pandas
from scipy.spatial.distance import cosine


class cruncher:
    def __init__(self):
        self.data = []

    def readcsv(self):
        dataset = pandas.read_csv('../data-files/usage_by_practitioner.csv').fillna(0)
        data_filter = dataset.drop('Unnamed: 0', 1)
        return data_filter

    def simcosine(self, data):
        #Item Based Recommendation
        data_frame = pandas.DataFrame(index=data.columns, columns=data.columns)
        for i in range(0, len(data_frame.columns)):
            for j in range(0, len(data_frame.columns)):
                data_frame.ix[i, j] = 1 - cosine(data.ix[:, i], data.ix[:, j])

        data_nb = pandas.DataFrame(index=data_frame.columns, columns=range(1,11))

        for i in range(0, len(data_frame.columns)):
            #print (data_nb.ix[i,:10].shape)
            #print (data_frame.ix[0:,i].order(ascending=False)[:10].index)
            data_nb.ix[i,:10] = data_frame.ix[0:,i].order(ascending=False)[:10].index

        return data_frame, data_nb

    def getScore(self, history, sim):
        return sum(history*sim)/sum(sim)

    def usercosinesim(self, data, data_frame, data_nb):
        data_sim = pandas.DataFrame(index=data.index, columns=data.columns)
        data_sim.ix[:,:1] = data.ix[:,:1]

        for i in range(0, len(data_sim.index)):
            for j in range(0, len(data_sim.columns)):
                user = data_sim.index[i]
                subtest = data_sim.columns[j]

                if data.ix[i][j] == 1:
                    data_sim.ix[i][j] = 0
                else:
                    subtest_top_names = data_nb.ix[subtest][1:10]
                    subtest_top_sim = data_frame.ix[subtest].order(ascending=False)[1:10]
                    user_top_subtest = data.ix[user, subtest_top_names]

                    data_sim.ix[i][j] = self.getScore(user_top_subtest, subtest_top_sim)

        return data_sim

    def recommender(self, data):
        data_rec = pandas.DataFrame(index=data.index, columns=['user', '1', '2', '3', '4', '5'])
        data_rec.ix[0:,0] = data.ix[:,0]

        for i in range(0, len(data.index)):
            data_rec.ix[i, 1:] = data[i,:].order(ascending=False).ix[1:7,]

        print (data_rec.ix[:10,:4])


def main():
    cr = cruncher()
    data = cr.readcsv()
    df, dnb = cr.simcosine(data)
    #data_sim = cr.usercosinesim(data, df, dnb)
    #cr.recommender(data_sim)
    df.to_csv('../data-files/item_sim.csv', sep='\t', encoding='utf-8')



if __name__ == '__main__':
    main()
