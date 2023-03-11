#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ConductSklearnStandardScaling:
    def __init__(self, data, *args, **kwargs):
        super(ConductSklearnStandardScaling, self).__init__(*args, **kwargs)#
        self.dim_x = data.dim_x
        self.num_cats = data.num_cats
        self.xtrain=data.xtrain
        self.xtest=data.xtest
        self.ytrain=data.ytrain
        self.ytest=data.ytest
        self.standardise()

    def standardise(self):
        x = self.xtrain
        x_columns = x.columns
        if 'master_idx' in x_columns:
            master_idx = x.master_idx
        if 'sess_idx' in x_columns:
            sess_idx = x.sess_idx
        if 'speaker_id' in x_columns:
            speaker_id = x.speaker_id
        if 'sess_id' in x_columns:
            sess_id = x.sess_id
        x = x.loc[:, x.columns != 'master_idx']
        x = x.loc[:, x.columns != 'sess_idx']
        x = x.loc[:, x.columns != 'speaker_id']
        x = x.loc[:, x.columns != 'sess_id']
        scaler = StandardScaler()
        scaler.fit(x)
        x = pd.DataFrame(scaler.transform(x))
        if 'sess_id' in x_columns:
            x = pd.concat([sess_id, x], axis=1)
        if 'speaker_id' in x_columns:
            x = pd.concat([speaker_id, x], axis=1)
        if 'sess_idx' in x_columns:
            x = pd.concat([sess_idx, x], axis=1)
        if 'master_idx' in x_columns:
            x = pd.concat([master_idx, x], axis=1)
        x.columns=x_columns
        self.xtrain = x
        x = self.xtest
        if 'master_idx' in x_columns:
            master_idx = x.master_idx
        if 'sess_idx' in x_columns:
            sess_idx = x.sess_idx
        if 'speaker_id' in x_columns:
            speaker_id = x.speaker_id
        if 'sess_id' in x_columns:
            sess_id = x.sess_id
        x = x.loc[:, x.columns != 'master_idx']
        x = x.loc[:, x.columns != 'sess_idx']
        x = x.loc[:, x.columns != 'speaker_id']
        x = x.loc[:, x.columns != 'sess_id']
        x = pd.DataFrame(scaler.transform(x))

        if 'sess_id' in x_columns:
            x = pd.concat([sess_id, x], axis=1)
        if 'speaker_id' in x_columns:
            x = pd.concat([speaker_id, x], axis=1)
        if 'sess_idx' in x_columns:
            x = pd.concat([sess_idx, x], axis=1)
        if 'master_idx' in x_columns:
            x = pd.concat([master_idx, x], axis=1)
        x.columns=x_columns
        self.xtest=x
        return self
