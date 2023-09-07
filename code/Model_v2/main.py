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
    initial_condition = [initial_space], #  Initial conditions
    n_epi = [4], # Number of epi updates between two decision updates
    beta = [0.1], # Infection rate
    r_base = [0.25], # basis contact rate
    r1 = [0.25], # Additional contact on diagonal entries
    r2 = [1], #  Additional contacts between two work risk groups
    r3 = [0.5], # Additional contacts between low-SES risk groups
    alpha = [0.02], # rate from recovered to susceptible
    gamma = [0.1], # recovery rate
    theta_x = [-0.5], # penalty for being sick
    theta_k = [1], # Extra sensitivity of vulnerable population
    theta_h = [1], # Sensitivity to hassle costs
    B = [1], # Payoff for making risky decision
    pc = [1], # Additional costs of risky behavior if infected
    mu = [0], # mean of log hassle cost
    sigma = [0.15], # std of log hassle cost
    threshold = [1e-5], # tolerance of solving Bellman function
    v = [0.57721], # Euler’s constant
    k = [0.9], # Discount factor
    c_l = [13], # Baseline level of consumption for low-SES
    c_h = [29], # Baseline level of consumption for high-SES
    w_l = [51], #  Wage for low-SES individuals
    w_h = [115] #  Wage for high-SES individuals
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
