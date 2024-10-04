from src import fiem

from collections import OrderedDict
from collections import namedtuple
from itertools import product
import pandas as pd
import numpy as np

risk_group_mapping = pd.DataFrame(columns=['Risk_group', 'Decision', 'State_variable_VUL', 'State_variable_SES'])
risk_group_mapping['Risk_group'] = ['g0-00', 'g0-01', 'g0-10', 'g0-11', 'g1-00', 'g1-01', 'g1-10', 'g1-11']
risk_group_mapping['Decision'] = ['not-work', 'not-work', 'not-work', 'not-work',
                                  'work', 'work', 'work', 'work']
risk_group_mapping['State_variable_VUL'] = ['not-vulnerable', 'not-vulnerable', 'vulnerable', 'vulnerable',
                                            'not-vulnerable', 'not-vulnerable', 'vulnerable', 'vulnerable']
risk_group_mapping['State_variable_SES'] = ['high-SES', 'low-SES', 'high-SES', 'low-SES',
                                            'high-SES', 'low-SES', 'high-SES', 'low-SES']

class RunBuilder:
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

#### Initialize population
initial_space1 = [
    {'amount': 1998, 'infection_status': 's', 'risk_group': 'g0-00'},
    {'amount': 1998, 'infection_status': 's', 'risk_group': 'g0-01'},
    {'amount': 1998, 'infection_status': 's', 'risk_group': 'g0-10'},
    {'amount': 1998, 'infection_status': 's', 'risk_group': 'g0-11'},
    {'amount': 1998, 'infection_status': 's', 'risk_group': 'g1-00'},
    {'amount': 1998, 'infection_status': 's', 'risk_group': 'g1-01'},
    {'amount': 1998, 'infection_status': 's', 'risk_group': 'g1-10'},
    {'amount': 1998, 'infection_status': 's', 'risk_group': 'g1-11'},
    {'amount': 2, 'infection_status': 'i', 'risk_group': 'g0-00'},
    {'amount': 2, 'infection_status': 'i', 'risk_group': 'g0-01'},
    {'amount': 2, 'infection_status': 'i', 'risk_group': 'g0-10'},
    {'amount': 2, 'infection_status': 'i', 'risk_group': 'g0-11'},
    {'amount': 2, 'infection_status': 'i', 'risk_group': 'g1-00'},
    {'amount': 2, 'infection_status': 'i', 'risk_group': 'g1-01'},
    {'amount': 2, 'infection_status': 'i', 'risk_group': 'g1-10'},
    {'amount': 2, 'infection_status': 'i', 'risk_group': 'g1-11'}
]

params = OrderedDict(
    initial_condition = [initial_space1], # Initial conditions
    n_epi = [4], # Number of epi updates between two decision updates
    beta = [0.2], # Infection rate
    r_base = [0.5], # basis contact rate
    r1 = [1.5], # Additional contact on same health-economic status
    r2 = [4], # Additional contacts between two work risk groups
    r3 = [1.5], # Additional contacts between low-SES risk groups
    alpha = [0.0043], # rate from recovered to susceptible
    gamma = [0.14], # recovery rate
    theta_x = [-5], # penalty for being sick
    theta_k = [3], # Extra sensitivity of vulnerable population
    theta_h = [0.5], # Sensitivity to hassle costs
    B = [0], # Payoff for making risky decision
    pc = [2], # Additional costs of risky behavior if infected
    mu = [0], # mean of log hassle cost
    sigma = [0.25], # std of log hassle cost
    threshold = [1e-5], # tolerance of solving Bellman function
    v = [0.57721], # Euler’s constant
    k = [0.96], # Discount factor
    c_l = [15], # Baseline level of consumption for low-SES
    c_h = [66], # Baseline level of consumption for high-SES
    w_l = [98], # Wage for low-SES individuals
    w_h = [262], # Wage for high-SES individuals
    lag = [7],
    policy_name = ['conditional'], # Policy name
    # work_prob = [0.7], # proportion of work after policy
    policy_start = [20], # start time of policy
    policy_end = [140], # end time of policy
    total_time = [140], # total time of simulation
    cash_transfer = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80], # cash transfer amount
    pay_sick = [90], # cash transfer amount
    how = ['random'] # how to choose people to be assign to not work for labor restriction policy
)

#### Direction to save results
file_forced_behavior = ''
file_unconditional = ''
file_conditional = ''
file_paid_sick_leave = ''
file_no_policy = ''



index = 0
num_simulations = 10 ### Number of simulations for each parameter set
dic_ref = []
for run in RunBuilder.get_runs(params):
    index += 1

    for i in np.arange(num_simulations):

        test = fiem.Model(initial_condition=run.initial_condition,
                     n_epi=run.n_epi,
                     beta=run.beta,
                     r_base=run.r_base,
                     r1=run.r1,
                     r2=run.r2,
                     r3=run.r3,
                     alpha=run.alpha,
                     gamma=run.gamma,
                     theta_x=run.theta_x,
                     theta_k=run.theta_k,
                     theta_h=run.theta_h,
                     B=run.B,
                     pc=run.pc,
                     mu=run.mu,
                     sigma=run.sigma,
                     threshold=run.threshold,
                     v=run.v,
                     k=run.k,
                     c_l=run.c_l,
                     c_h=run.c_h,
                     w_l=run.w_l,
                     w_h=run.w_h,
                     pay_sick = 0,
                     )
        
        if run.policy_name == 'forced_behavior':
            df = test.run_forced_behavior_hybrid(run.policy_start, 
                                        run.policy_end,
                                        run.total_time, 
                                        run.work_prob, 
                                        how = run.how,
                                        lag=run.lag)
            
            df = df.drop(columns=['index'])
            df['count'] = 1
            df = df[['time', 'infection_status', 'risk_group', 'count']]
            df = df.groupby(['time', 'infection_status', 'risk_group']).sum()
            df = df.reset_index()
            df.to_csv(file_forced_behavior + '/model_' + str(run.work_prob) + '_run_' + str(i + 1) + '.csv')
        if run.policy_name == 'unconditional':
            # df = test.run_cash_transfer(start_time, end_time, total_iteration, cash_transfer, how)
            df = test.run_cash_transfer(run.policy_start, 
                                        run.policy_end,
                                        run.total_time, 
                                        run.cash_transfer,
                                        how = run.policy_name,
                                        lag=run.lag)
            
            df = df.drop(columns=['index'])
            df['count'] = 1
            df = df[['time', 'infection_status', 'risk_group', 'count']]
            df = df.groupby(['time', 'infection_status', 'risk_group']).sum()
            df = df.reset_index()
            df.to_csv(file_unconditional + '/model_' + str(run.cash_transfer) + '_run_' + str(i + 1) + '.csv')
        if run.policy_name == 'conditional':
            # df = test.run_cash_transfer(start_time, end_time, total_iteration, cash_transfer, how)
            df = test.run_cash_transfer(run.policy_start, 
                                        run.policy_end,
                                        run.total_time, 
                                        run.cash_transfer,
                                        how = run.policy_name,
                                        lag=run.lag)
            
            df = df.drop(columns=['index'])
            df['count'] = 1
            df = df[['time', 'infection_status', 'risk_group', 'count']]
            df = df.groupby(['time', 'infection_status', 'risk_group']).sum()
            df = df.reset_index()
            df.to_csv(file_conditional + '/model_' + str(run.cash_transfer) + '_run_' + str(i + 1) + '.csv')
            
        if run.policy_name == 'paid_sick_leave':
            df = test.run_paid_sick_leave(run.policy_start, 
                                        run.policy_end,
                                        run.total_time, 
                                        run.pay_sick,
                                        lag = run.lag)
            df = df.drop(columns=['index'])
            df['count'] = 1
            df = df[['time', 'infection_status', 'risk_group', 'count']]
            df = df.groupby(['time', 'infection_status', 'risk_group']).sum()
            df = df.reset_index()
            df.to_csv(file_paid_sick_leave + '/model_' + str(run.pay_sick) + '_run_' + str(i + 1) + '.csv')
        if run.policy_name == 'no_policy':
            df = test.run_econ_epi_lag(365, run.lag)
            df['count'] = 1
            df = df[['time', 'infection_status', 'risk_group', 'count']]
            df = df.groupby(['time', 'infection_status', 'risk_group']).sum()
            df = df.reset_index()
            df.to_csv(file_no_policy + '/model_' + str(run.gamma) + '_run_' + str(i + 2) + '.csv')
        