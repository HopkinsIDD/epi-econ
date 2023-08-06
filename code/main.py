import pandas as pd
import numpy as np

from Model import Model, Agent
from collections import OrderedDict
from collections import namedtuple
from itertools import product

risk_group_mapping = pd.DataFrame(columns = ['Risk_group', 'Decision', 'State_variable'])
risk_group_mapping['Risk_group'] = ['g1', 'g2', 'g3', 'g4']
risk_group_mapping['Decision'] = ['risky', 'non-risky', 'risky', 'non-risky']
risk_group_mapping['State_variable'] = ['vulnerable', 'vulnerable', 'non-vulnerable', 'non-vulnerable']

file = 'Model_updated_v6/n_epi' # path to save simulation results

initial_space = [
    {'amount' : 5000, 'infection_status' : 's','risk_group' : 'g1'},
    {'amount' : 5000, 'infection_status' : 's','risk_group' : 'g2'},
    {'amount' : 5000, 'infection_status' : 's','risk_group' : 'g3'},
    {'amount' : 5000, 'infection_status' : 's','risk_group' : 'g4'},
    {'amount' : 5,  'infection_status' : 'i','risk_group' : 'g1'},
    {'amount' : 5, 'infection_status' : 'i', 'risk_group' : 'g2'},
    {'amount' : 5, 'infection_status' : 'i','risk_group' : 'g3'},
    {'amount' : 5, 'infection_status' : 'i', 'risk_group' : 'g4'}
    ]

params = OrderedDict(
    initial_condition = [initial_space],
    n_epi = [1, 2, 4, 8, 10],
    beta = [0.02, 0.025, 0.03, 0.035, 0.04],
    C_ij=[np.array([[2, 2, 2, 2],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [1, 1, 1, 1]])],
    phi_i = [np.array([[1.5], [1.5], [1], [1]])],
    alpha = [0.02],
    gamma = [0.1],
    theta_x = [-1],
    theta_k = [0.5],
    theta_h = [0.5],
    B = [1],
    pc = [0.5],
    mu = [0],
    sigma = [0.25],
    threshold = [1e-5],
    v = [0.57721],
    k = [0.9]
)

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

index = 1
num_simulations = 10
dic_ref = []
for run in RunBuilder.get_runs(params):

    for i in np.arange(num_simulations):

        test = Model(initial_condition = run.initial_condition,
                     n_epi = run.n_epi,
                     beta = run.beta,
                     C_ij = run.C_ij,
                     phi_i = run.phi_i,
                     alpha = run.alpha,
                     gamma = run.gamma,
                     theta_x = run.theta_x,
                     theta_k = run.theta_k,
                     theta_h = run.theta_h,
                     B = run.B,
                     pc = run.pc,
                     mu = run.mu,
                     sigma = run.sigma,
                     threshold = run.threshold,
                     v = run.v,
                     k = run.v)

        df = test.run(100)

        dic = {
            'Model' : str(index),
            'initial_condition' : run.initial_condition,
            'n_epi' : run.n_epi,
            'beta' : run.beta,
            'C_ij' : run.C_ij,
            'phi_i' : run.phi_i,
            'alpha' : run.alpha,
            'gamma' : run.gamma,
            'theta_x' : run.theta_x,
            'theta_k' : run.theta_k,
            'theta_h' : run.theta_h,
            'B' : run.B,
            'pc' : run.pc,
            'mu' : run.mu,
            'sigma' : run.sigma,
            'threshold' : run.threshold,
            'v' : run.v,
            'k' : run.k,
            'Model_run' : i + 1
        }
        dic_ref.append(dic)

        df.to_pickle(file + '/model_' + str(index) + '_run_' + str(i+1) + '.pkl')

    index += 1

df_ref = pd.DataFrame(dic_ref)
df_ref.to_pickle(file + '/model_ref.pkl')