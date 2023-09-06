from collections import OrderedDict
from collections import namedtuple
from itertools import product
from Model import Model
# Manage parameters for simulation


class RunBuilder:
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


initial_space = [
    {'amount': 500, 'infection_status': 's', 'risk_group': 'g0-00'},
    {'amount': 500, 'infection_status': 's', 'risk_group': 'g0-01'},
    {'amount': 500, 'infection_status': 's', 'risk_group': 'g0-10'},
    {'amount': 500, 'infection_status': 's', 'risk_group': 'g0-11'},
    {'amount': 500, 'infection_status': 's', 'risk_group': 'g1-00'},
    {'amount': 500, 'infection_status': 's', 'risk_group': 'g1-01'},
    {'amount': 500, 'infection_status': 's', 'risk_group': 'g1-10'},
    {'amount': 500, 'infection_status': 's', 'risk_group': 'g1-11'},
    {'amount': 1, 'infection_status': 'i', 'risk_group': 'g0-00'},
    {'amount': 1, 'infection_status': 'i', 'risk_group': 'g0-01'},
    {'amount': 1, 'infection_status': 'i', 'risk_group': 'g0-10'},
    {'amount': 1, 'infection_status': 'i', 'risk_group': 'g0-11'},
    {'amount': 1, 'infection_status': 'i', 'risk_group': 'g1-00'},
    {'amount': 1, 'infection_status': 'i', 'risk_group': 'g1-10'},
    {'amount': 1, 'infection_status': 'i', 'risk_group': 'g1-01'},
    {'amount': 1, 'infection_status': 'i', 'risk_group': 'g1-11'}

]

# Sensitivity analysis with different parameters

params = OrderedDict(
    initial_condition = [initial_space],
    n_epi = [4],
    beta = [0.1],
    r_base = [0.2],
    r1 = [0.5],
    r2 = [1],
    r3 = [0.5],
    alpha = [0.02],
    gamma = [0.1],
    theta_x = [-0.5],
    theta_k = [1],
    theta_h = [1],
    B = [1],
    pc = [1],
    mu = [0],
    sigma = [0.15],
    threshold = [1e-5],
    v = [0.57721],
    k = [0.9],
    c_l = [13],
    c_h = [29],
    w_l = [51],
    w_h = [115]
)

for run in RunBuilder.get_runs(params):
    test_1 = Model(initial_condition=run.initial_condition,
                   n_epi = run.n_epi,
                   beta = run.beta,
                   r_base = run.r_base,
                   r1 = run.r1,
                   r2 = run.r2,
                   r3 = run.r3,
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
                   k = run.k,
                   c_l = run.c_l,
                   c_h = run.c_h,
                   w_l = run.w_l,
                   w_h = run.w_h)

df_1 = test_1.run(5)
