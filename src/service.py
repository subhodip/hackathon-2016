#!/usr/bin/env python
from json_tricks import dumps, np, nonp
import logging
from wsgiref import simple_server
from pearsoncoer import pcr
import falcon
import numpy as nump


class GetRecommendation():
    def get_recommend(self, sub_idx):
        k_array = [5, 15, 30, 50, 100, 200]
        user_train_mse = []
        user_test_mse = []
        item_test_mse = []
        item_train_mse = []
        rec_subtest = []
        p = pcr()
        dfs, orig_data = p.sparsity()
        train, test = p.train_test_split(dfs)
        item_sim = p.fast_sim(train, kind='item')
        #user_sim = p.fast_sim(train, kind='user')
        #user_predict = p.predict_fast_simple(train, user_sim, kind='user')
        #u_pred_topk = p.predict_topk(train, user_sim, kind='user', k=40)
        #i_pred_topk = p.predict_topk(train, item_sim, kind='item', k=40)
        #print('Top-k User-based CF MSE: ' + str(p.get_mse(u_pred_topk, test.as_matrix())))
        #print('Top-k Item-based CF MSE: ' + str(p.get_mse(i_pred_topk, test.as_matrix())))
        # print (u_pred_topk)
        l = p.top_k_users(item_sim, dfs, idx=sub_idx)
        # u = p.top_k_users(user_sim, dfs, idx=0)
        # print (u)
        for isbn in l:
            rec_subtest.append(orig_data.loc[orig_data['ISBN'] == isbn]['Subtest'].unique().tolist())

        # for k in k_array:
        #    user_pred = p.predict_topk(train, user_sim, kind='user', k=k)
        #    item_pred = p.predict_topk(train, item_sim, kind='item', k=k)

        #    user_train_mse += [p.get_mse(user_pred, train.as_matrix())]
        #    user_test_mse += [p.get_mse(user_pred, test.as_matrix())]

        #    item_train_mse += [p.get_mse(item_pred, train.as_matrix())]
        #    item_test_mse += [p.get_mse(item_pred, test.as_matrix())]

        return rec_subtest


class ThingsResource(object):
    def on_get(self, req, resp):
        query = req.query_string
        resp.status = falcon.HTTP_200
        sl = GetRecommendation().get_recommend(int(query))
        resp.body = np.dumps({'subtest': sl[0], 'recommendation': sl[1:]})


wsgi_app = api = falcon.API()

service = ThingsResource()

api.add_route('/recommend', service)
