#!/usr/bin/env python
# A service will sent the crunch data to this module. This module will generate both the euclidean distance and
# pearson correlation and associate a rank to the subtest.
from scipy.spatial.distance import cosine
import pandas


class Recommend:
    def readcsv(self):
        dataset = pandas.read_csv('../data-files/usage_by_practitioner.csv').fillna(0)
        return dataset

    def computesim(self, dataset):
        data = dataset.drop('Unnamed: 0', 1)
        data_frame = pandas.DataFrame(index=data.columns, columns=data.columns)
        for i in range(0, len(data_frame.columns)):
            for j in range(0, len(data_frame.columns)):
                data_frame.ix[i, j] = 1 - cosine(data.ix[:, i], data.ix[:, j])

        data_nb = pandas.DataFrame(index=data_frame.columns, columns=range(1, 11))

        for i in range(0, len(data_frame.columns)):
            # print (data_nb.ix[i,:10].shape)
            # print (data_frame.ix[0:,i].order(ascending=False)[:10].index)
            data_nb.ix[i, :10] = data_frame.ix[0:, i].order(ascending=False)[:10].index
        return dataset, data_frame, data_nb

    def getScore(self, his, sim):
        return sum(his * sim) / sum(sim)

    def usersim(self, data, data_frame, data_nb):
        data_filter = data.drop('Unnamed: 0', 1)
        data_sim = pandas.DataFrame(index=data.index, columns=data.columns)
        data_sim.ix[:, :1] = data.ix[:, :1]

        for i in range(0, len(data_sim.index)):
            for j in range(1, len(data_sim.columns)):
                user = data_sim.index[i]
                subtest = data_sim.columns[j]

                if data.ix[i][j] == 1:
                    data_sim.ix[i][j] = 0
                else:
                    subtest_top_isbn = data_nb.ix[subtest][1:11]
                    subtest_top_sim = data_frame.ix[subtest].sort_values(ascending=False)[1:11]
                    usage = data_filter.ix[user, subtest_top_isbn]
                    data_sim.ix[i][j] = self.getScore(usage, subtest_top_sim)

        data_rec = pandas.DataFrame(index=data_sim.index, columns=['user', '1', '2', '3', '4', '5', '6'])
        data_rec.ix[0:, 0] = data_sim.ix[:, 0]

        for i in range(0, len(data_sim.index)):
            data_rec.ix[i, 1:] = data_sim.ix[i, :].order(ascending=False).ix[1:7, ].index.transpose()

        return data_rec


def main():
    r = Recommend()
    data = r.readcsv()
    dataset, data_frame, data_nb = r.computesim(data)
    dr = r.usersim(data, data_frame, data_nb)
    dr.to_csv('../data-files/user_sim.csv', sep='\t', encoding='utf-8')


if __name__ == '__main__':
    main()
