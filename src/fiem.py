import numpy as np
import random
from itertools import product
from collections import defaultdict
import pandas as pd
from tqdm import trange

risk_group_mapping = pd.DataFrame(columns=['Risk_group', 'Decision', 'State_variable_VUL', 'State_variable_SES'])
risk_group_mapping['Risk_group'] = ['g0-00', 'g0-01', 'g0-10', 'g0-11', 'g1-00', 'g1-01', 'g1-10', 'g1-11']
risk_group_mapping['Decision'] = ['not-work', 'not-work', 'not-work', 'not-work',
                                  'work', 'work', 'work', 'work']
risk_group_mapping['State_variable_VUL'] = ['not-vulnerable', 'not-vulnerable', 'vulnerable', 'vulnerable',
                                            'not-vulnerable', 'not-vulnerable', 'vulnerable', 'vulnerable']
risk_group_mapping['State_variable_SES'] = ['high-SES', 'low-SES', 'high-SES', 'low-SES',
                                            'high-SES', 'low-SES', 'high-SES', 'low-SES']

class Agent:
    """
    Create each individual with different attributes

    Attributes:
        index: the index for each individual
        time: running time
        decision: risky or non-risk
        state_VUL: state variable health
        state_SES: state variable econ
        infection_status: s, i, r
        risk_group: depends on state and infection_status
        hassle_cost_group: low, median, high
    """

    RISK_GROUP_MAPPING = {
        'g0-00': ('not-work', 'not-vulnerable', 'high-SES'),
        'g0-01': ('not-work', 'not-vulnerable', 'low-SES'),
        'g0-10': ('not-work', 'vulnerable', 'high-SES'),
        'g0-11': ('not-work', 'vulnerable', 'low-SES'),
        'g1-00': ('work', 'not-vulnerable', 'high-SES'),
        'g1-01': ('work', 'not-vulnerable', 'low-SES'),
        'g1-10': ('work', 'vulnerable', 'high-SES'),
        'g1-11': ('work', 'vulnerable', 'low-SES'),
    }

    def __init__(self, m, t, x, g):
        self.index = m
        self.time = t
        self.infection_status = x
        self.risk_group = g

        self.decision, self.state_VUL, self.state_SES = self.RISK_GROUP_MAPPING[self.risk_group]

        # initialize hassle cost label, will randomly assign to
        # three groups later
        self.hassle_cost_group = 'low'

    def __repr__(self):
        """
        String representation of Agent
        """
        return 'Agent with index = {}'.format(self.index) + \
            ', time = {}'.format(self.time) + \
            ', risk group = {}'.format(self.risk_group) + \
            ', infection status = {}'.format(self.infection_status)

    def update_time(self):
        """update running time"""
        self.time = self.time + 1

    def update_risk_group(self):
        """update risk group based on state and decisions"""
        risk_group_mapping = {
            ('not-vulnerable', 'high-SES'): '00',
            ('not-vulnerable', 'low-SES'): '01',
            ('vulnerable', 'high-SES'): '10',
            ('vulnerable', 'low-SES'): '11',
        }

        decision_code = '1' if self.decision == 'work' else '0'
        risk_group_code = risk_group_mapping.get((self.state_VUL, self.state_SES), '00')

        self.risk_group = f'g{decision_code}-{risk_group_code}'


    def update_decision(self, p_decision):
        """update decision based on output from Bellman function
        ----------
        Parameters
        p_decision: probability decision matrix

        ----------
        Outputs
        Update decision and risk group
        """
        if self.infection_status == 's':
            p = p_decision.iloc[0, 0]
        elif self.infection_status == 'i':
            p = p_decision.iloc[1, 0]
        else:
            p = p_decision.iloc[2, 0]
        rv = np.random.uniform(0, 1, 1).item()
        if rv < p:
            self.decision = 'work'
        else:
            self.decision = 'not-work'

        if self.state_VUL == 'not-vulnerable' and self.state_SES == 'high-SES':
            if self.decision == 'not-work':
                self.risk_group = 'g0-00'
            else:
                self.risk_group = 'g1-00'

        if self.state_VUL == 'not-vulnerable' and self.state_SES == 'low-SES':
            if self.decision == 'not-work':
                self.risk_group = 'g0-01'
            else:
                self.risk_group = 'g1-01'

        if self.state_VUL == 'vulnerable' and self.state_SES == 'high-SES':
            if self.decision == 'not-work':
                self.risk_group = 'g0-10'
            else:
                self.risk_group = 'g1-10'

        if self.state_VUL == 'vulnerable' and self.state_SES == 'low-SES':
            if self.decision == 'not-work':
                self.risk_group = 'g0-11'
            else:
                self.risk_group = 'g1-11'

    def update_infection_status(self, p):
        """update infection status based on transition matrix
        ----------
        Parameters
        p: transition probability matrix

        ----------
        Output
        update infection status
        """
        rv = np.random.uniform(0, 1, 1).item()
        if self.infection_status == 's':
            if rv <= p.iloc[0, 1]:
                self.infection_status = 'i'
            else:
                self.infection_status = 's'

        elif self.infection_status == 'i':
            if rv <= p.iloc[1, 2]:
                self.infection_status = 'r'
            else:
                self.infection_status = 'i'

        elif self.infection_status == 'r':
            if rv <= p.iloc[2, 0]:
                self.infection_status = 's'
            else:
                self.infection_status = 'r'


### FIEM Model
class Model:
    '''
    Model class
    
    Attributes:
    initial_condition: list of dictionary, each dictionary contains the initial condition of the agent
    n_epi: number of iteration for epi model
    beta: infection rate
    r_base: base contact rate
    r1: contact rate for the same group
    r2: contact rate for the same group and same decision
    r3: contact rate for the same group and same decision and same infection status
    alpha: waning rate
    gamma: recovery rate
    theta_x: utility cost of infetion
    theta_k: utility cost for vulnearble 
    theta_h: utility cost for hassle cost
    B: utility benefit for working
    pc: additional utility cost for working if infected
    mu: mean of the log-normal distribution for hassle cost
    sigma: standard deviation of the log-normal distribution for hassle cost
    threshold: threshold for the Bellman function convergence
    v: constant for the Bellman function
    k: discounting factor
    c_l: baseline consumption for low SES
    c_h: baseline consumption for high SES
    w_l: wage for low SES
    W_h: wage for high SES
    pay_sick: payment for sick leave (only take nonzero values when apply to the paid sick leave policy)
    '''
    def __init__(self, initial_condition, n_epi, beta, r_base, r1, r2, r3,
                 alpha, gamma, theta_x, theta_k, theta_h, B, pc,
                 mu=0, sigma=0.5, threshold=1e-5, v=0.57721, k=0.9,
                 c_l=13, c_h=29, w_l=51, w_h=115, pay_sick = 0):
        self.iteration = None
        self.total_agent = []
        for spec in initial_condition:
            for i in range(spec['amount']):
                new_agent = Agent(len(self.total_agent), 0, spec['infection_status'],
                                  spec['risk_group'])
                self.total_agent.append(new_agent)

        index = np.arange(len(self.total_agent))
        random.seed(1)
        random.shuffle(index)

        for i in range(round(len(self.total_agent) / 3)):
            self.total_agent[index[i]].hassle_cost_group = 'low'
        for i in np.arange(round(len(self.total_agent) / 3), 2 * round(len(self.total_agent) / 3)):
            self.total_agent[index[i]].hassle_cost_group = 'median'
        for i in np.arange(2 * round(len(self.total_agent) / 3), len(self.total_agent)):
            self.total_agent[index[i]].hassle_cost_group = 'high'

        self.n_epi = n_epi
        self.beta = beta
        self.r_base = r_base
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3

        """
        Define Contact matrix here:
        """

        self.contact_matrix = pd.DataFrame(np.array(np.zeros([8, 8])),
                                           index=['g0-00', 'g0-01', 'g0-10', 'g0-11', 'g1-00', 'g1-01', 'g1-10',
                                                  'g1-11'],
                                           columns=['g0-00', 'g0-01', 'g0-10', 'g0-11', 'g1-00', 'g1-01', 'g1-10',
                                                    'g1-11'])

        self.contact_matrix = self.contact_matrix + self.r_base

        for row in self.contact_matrix.index:
            for col in self.contact_matrix.columns:
                if row[-2:] == col[-2:]:
                    self.contact_matrix.loc[row, col] = self.contact_matrix.loc[row, col] * self.r1
                if (row[1] == '1') & (col[1] == '1'):
                    self.contact_matrix.loc[row, col] = self.contact_matrix.loc[row, col] * self.r2
                if (row[4] == '1') & (col[4] == '1'):
                    self.contact_matrix.loc[row, col] = self.contact_matrix.loc[row, col] * self.r3

        self.Beta = self.beta * self.contact_matrix

        self.alpha = pd.DataFrame(np.array([alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha]),
                                  index=['g0-00', 'g0-01', 'g0-10', 'g0-11', 'g1-00', 'g1-01', 'g1-10', 'g1-11'],
                                  columns=['alpha'])

        self.gamma = pd.DataFrame(np.array([gamma, gamma, gamma, gamma, gamma, gamma, gamma, gamma]),
                                  index=['g0-00', 'g0-01', 'g0-10', 'g0-11', 'g1-00', 'g1-01', 'g1-10', 'g1-11'],
                                  columns=['gamma'])

        self.theta_x = theta_x
        self.theta_k = theta_k
        self.theta_h = theta_h

        self.B = B
        self.pc = pc
        self.mu = mu
        self.sigma = sigma

        self.threshold = threshold
        self.v = v
        self.k = k
        self.c_l = c_l
        self.c_h = c_h
        self.w_l = w_l
        self.w_h = w_h

        self.pay_sick = pay_sick
        
    def init_transition_matrix(self, risk_group, n):
        '''
        Initialize the transition matrix for each risk group
        '''
        
        p_si = 0
        
        p_ss = 1 - p_si
        
        p_ir = self.gamma.loc[risk_group].item() / n

        p_ii = 1 - p_ir

        p_rs = self.alpha.loc[risk_group].item() / n

        p_rr = 1 - p_rs

        p = pd.DataFrame(np.array([[p_ss, p_si, 0],
                                   [0, p_ii, p_ir],
                                   [p_rs, 0, p_rr]]),
                         index=['s', 'i', 'r'],
                         columns=['s', 'i', 'r'])
        return p
        
        

    def transition_matrix(self, risk_group, n):
        '''
        Transition matrix for each risk group
        risk_group: risk group
        n: number of epi updates
        '''
        p_si = 0
        N = len(self.total_agent)
        for g in {agent.risk_group for agent in self.total_agent}:
            I_g = sum((agent.risk_group == g) & (agent.infection_status == "i") for agent in self.total_agent)
            p_si += (I_g * self.Beta.loc[risk_group, g] / n) / N
        if p_si >= 1:
            p_si = 1

        p_ss = 1 - p_si

        p_ir = self.gamma.loc[risk_group].item() / n

        p_ii = 1 - p_ir

        p_rs = self.alpha.loc[risk_group].item() / n

        p_rr = 1 - p_rs

        p = pd.DataFrame(np.array([[p_ss, p_si, 0],
                                   [0, p_ii, p_ir],
                                   [p_rs, 0, p_rr]]),
                         index=['s', 'i', 'r'],
                         columns=['s', 'i', 'r'])
        return p
    
    def utility_func(self, infection_status, state_VUL, state_SES, decision, hassle_cost_group):
        '''
        Utility function
        infection_status: S, or I, or R
        state_VUL: vulnerable or not-vulnerable
        state_SES: low-SES or high-SES
        decision: work or not-work
        hassle_cost_group: low, median, high
        '''
        if hassle_cost_group == 'low':
            h = np.exp(-0.38)
        elif hassle_cost_group == 'median':
            h = np.exp(0)
        else:
            h = np.exp(0.38)
            
        if infection_status == 'i':
            x = 2
        elif infection_status == 'r':
            x = 1
        else:
            x = 0   
        
        if state_VUL == 'vulnerable':
            k = 1
        else:
            k = 0

        if state_SES == 'low-SES':
            c_base = self.c_l
            w = self.w_l
        else:
            c_base = self.c_h
            w = self.w_h
            
        if (x != 2) & (decision == 'work'):
            U = np.log(w) + self.B - self.theta_h * h
        if (x != 2) & (decision == 'not-work'):
            U = np.log(c_base)
            
        if (x == 2) & (decision == 'work'):
            U = np.log(w) + self.B + k*self.theta_k*self.theta_x + (1 - k)*self.theta_x - (1 + self.pc)*self.theta_h*h
        if (x == 2) & (decision == 'not-work'):
            U = np.log(c_base + self.pay_sick) + k*self.theta_k*self.theta_x + (1 - k)*self.theta_x
            
        return np.float64(U).item()

    def bellman_func(self, state_VUL, state_SES, hassle_cost_group,
                     p_g0_00, p_g0_01, p_g0_10, p_g0_11,
                     p_g1_00, p_g1_01, p_g1_10, p_g1_11):
        '''
        Bellman function, given the state variables, and transition matrices, solve for Bellman equation
        '''

        V = np.ones((3, 2))
        U = np.zeros((3, 2))
        V_new = np.ones((3, 2))

        error = 1
        i = 0
        for decision in ['work', 'not-work']:
            u = []
            for infection_status in ['s', 'i', 'r']:
                u.append(self.utility_func(infection_status, state_VUL, state_SES, decision, hassle_cost_group))
            U[:, i] = u
            i += 1

        if state_VUL == 'not-vulnerable' and state_SES == 'high-SES':
            p1 = p_g1_00
            p2 = p_g0_00
        elif state_VUL == 'not-vulnerable' and state_SES == 'low-SES':
            p1 = p_g1_01
            p2 = p_g0_01
        elif state_VUL == 'vulnerable' and state_SES == 'high-SES':
            p1 = p_g1_10
            p2 = p_g0_10
        # else state_VUL == 'vulnerable' and state_SES == 'low-SES':
        else:
            p1 = p_g1_11
            p2 = p_g0_11

        j = 0
        while error > self.threshold:
            V_new[:, 0] = U[:, 0] + self.k * p1.to_numpy().dot((np.log(np.exp(V).sum(axis=1)) + self.v))
            V_new[:, 1] = U[:, 1] + self.k * p2.to_numpy().dot((np.log(np.exp(V).sum(axis=1)) + self.v))
            error = abs(V - V_new).sum()
            V = V_new.copy()
            j += 1
            if j > 10000:
                print('Convergence failed')
                break

        p_decision = pd.DataFrame((np.exp(V) / np.exp(V).sum(axis=1)[:, None]),
                                  index=['s', 'i', 'r'],
                                  columns=['work', 'not-work'])

        return p_decision
    
    def decision_model(self, transition_matrices):

        unique_states = list(product(risk_group_mapping['State_variable_VUL'].unique(),
                                 risk_group_mapping['State_variable_SES'].unique(),
                                 ['low', 'median', 'high']))

        DDPs_list = {state: self.bellman_func(*state, *transition_matrices.values()) for state in unique_states}

        for agent in self.total_agent:
            agent.update_decision(DDPs_list[agent.state_VUL, agent.state_SES, agent.hassle_cost_group])

    def epi_model(self):
        '''
        Epi model
        '''
        for t in range(self.n_epi):

            transition_matrix_list = {}
            for rg in risk_group_mapping['Risk_group'].unique():
                transition_matrix_list[rg] = self.transition_matrix(rg, self.n_epi)

            '''
            Update infection
            '''
            for rg in risk_group_mapping['Risk_group'].unique():
                rg_S = [agent for agent in self.total_agent if agent.risk_group == rg
                        and agent.infection_status == 's']
                rg_I = [agent for agent in self.total_agent if agent.risk_group == rg
                        and agent.infection_status == 'i']
                rg_R = [agent for agent in self.total_agent if agent.risk_group == rg
                        and agent.infection_status == 'r']

                # generate random vector
                random_floats_S = np.random.rand(len(rg_S))
                random_floats_I = np.random.rand(len(rg_I))
                random_floats_R = np.random.rand(len(rg_R))

                indices_S_to_I = np.where(random_floats_S <= transition_matrix_list[rg].iloc[0, 1])[0]
                indices_I_to_R = np.where(random_floats_I <= transition_matrix_list[rg].iloc[1, 2])[0]
                indices_R_to_S = np.where(random_floats_R <= transition_matrix_list[rg].iloc[2, 0])[0]


                for j in indices_S_to_I:
                    rg_S[j].infection_status = 'i'
                for j in indices_I_to_R:
                    rg_I[j].infection_status = 'r'
                for j in indices_R_to_S:
                    rg_R[j].infection_status = 's'
    
    def run_econ_epi_lag(self, iteration, lag):
        ### Add lag
        self.iteration = iteration
        matrix = {}
        transition_matrices_t0 = {f'g{i}-{j}{k}': self.init_transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
        matrix[0] = transition_matrices_t0
        
        self.decision_model(transition_matrices_t0)
        dic_list = []
        for agent in self.total_agent:
            dic_list.append(agent.__dict__.copy())
        
        for i in trange(self.iteration):
            if i < lag:
                self.epi_model()
                
                transition_matrices = {f'g{i}-{j}{k}': self.transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
                matrix[i+1] = transition_matrices
                
                self.decision_model(transition_matrices_t0)
                
                for agent in self.total_agent:
                    agent.update_time()
                    dic_list.append(agent.__dict__.copy())
            
            else:
                self.epi_model()
                transition_matrices = {f'g{i}-{j}{k}': self.transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
                
                matrix[i+1] = transition_matrices
                
                self.decision_model(matrix[i+1 - lag])
                
                for agent in self.total_agent:
                    agent.update_time()
                    dic_list.append(agent.__dict__.copy())
                    
        return pd.DataFrame(dic_list)

    def run_econ_epi(self, iteration):
        self.iteration = iteration
        self.decision_model()

        dic_list = []
        for agent in self.total_agent:
            dic_list.append(agent.__dict__.copy())

        for _ in trange(self.iteration):
            self.epi_model()
            self.decision_model()

            for agent in self.total_agent:
                agent.update_time()
                dic_list.append(agent.__dict__.copy())

        return pd.DataFrame(dic_list)
    
    def run_epi_only(self, iteration):

        self.iteration = iteration

        dic_list = []
        for agent in self.total_agent:
            dic_list.append(agent.__dict__.copy())

        for _ in trange(self.iteration):
            self.epi_model()

            for agent in self.total_agent:
                agent.update_time()
                dic_list.append(agent.__dict__.copy())

        return pd.DataFrame(dic_list)
    
    ### Forced behavior policy

    def run_forced_behavior(self, start_time, end_time, total_iteration, proportion_work, how):
        self.inital_iteration = start_time
        self.decision_model()

        dic_list = []
        for agent in self.total_agent:
            dic_list.append(agent.__dict__.copy())

        for _ in trange(self.inital_iteration):
            self.epi_model()
            self.decision_model()

            for agent in self.total_agent:
                agent.update_time()
                dic_list.append(agent.__dict__.copy())

        if how == 'random':
            random.seed(1)
            selected_agents = np.random.choice(self.total_agent, size=int(len(self.total_agent) * proportion_work), replace=False)
            selected_agents_set = set(selected_agents)

            for agent in self.total_agent:
                if agent in selected_agents_set:
                    agent.decision = 'work'
                    agent.update_risk_group()
                else:
                    agent.decision = 'not work'
                    agent.update_risk_group()

        elif how == 'equal':
            # equal here means every risk group has the same proportion of agents working
            agents_by_risk_group = defaultdict(list)
            for agent in self.total_agent:
                agents_by_risk_group[agent.risk_group].append(agent)

        # For each risk group, randomly select 20% of the agents to set their decision to 'work'
        # and the rest to 'not work'
            for agents in agents_by_risk_group.values():
                random.seed(42)
                np.random.shuffle(agents)
                split_index = int(len(agents) * proportion_work)

                for i, agent in enumerate(agents):
                    if i < split_index:
                        agent.decision = 'work'
                        agent.update_risk_group()
                    else:
                        agent.decision = 'not work'
                        agent.update_risk_group()
        
        for _ in trange(end_time- self.inital_iteration):
            self.epi_model()

            for agent in self.total_agent:
                agent.update_time()
                dic_list.append(agent.__dict__.copy())
        
        for _ in trange(total_iteration - end_time):
            self.decision_model()
            self.epi_model()

            for agent in self.total_agent:
                agent.update_time()
                dic_list.append(agent.__dict__.copy())

        return pd.DataFrame(dic_list)
    
    #### This function is fore a selected amount of people to not work
    #### Rest of people can still work or not work based on the decision model
    def run_forced_behavior_hybrid(self, start_time, end_time, total_iteration, proportion_work, how, lag):
        self.inital_iteration = start_time
        
        matrix = {}
        transition_matrices_t0 = {f'g{i}-{j}{k}': self.init_transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
        matrix[0] = transition_matrices_t0
    
        self.decision_model(transition_matrices_t0)
        dic_list = []
        
        for agent in self.total_agent:
            dic_list.append(agent.__dict__.copy())

        for t in trange(self.inital_iteration):
            if t < lag:  
                self.epi_model()
                transition_matrices = {f'g{i}-{j}{k}': self.transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
                matrix[t+1] = transition_matrices
                
                self.decision_model(transition_matrices_t0)

                for agent in self.total_agent:
                    agent.update_time()
                    dic_list.append(agent.__dict__.copy())   
                    
            else:
                self.epi_model()
                transition_matrices = {f'g{i}-{j}{k}': self.transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
                
                matrix[t+1] = transition_matrices
                # print(t+1)
                
                self.decision_model(matrix[t+1 - lag])
                
                for agent in self.total_agent:
                    agent.update_time()
                    dic_list.append(agent.__dict__.copy())
                

        if how == 'random':
                random.seed(42)
                selected_agents = np.random.choice(self.total_agent, size=int(len(self.total_agent) * proportion_work), replace=False)
                selected_agents_set = set(selected_agents)

        for t in trange(end_time - self.inital_iteration):
            transition_matrices = {f'g{i}-{j}{k}': self.transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
            matrix[t+1 + self.inital_iteration] = transition_matrices
            self.decision_model(matrix[t+1 + self.inital_iteration - lag])
            # print(t+1+ self.inital_iteration)
                    
            for agent in self.total_agent:
                if agent not in selected_agents_set:
                    agent.decision = 'not work'
                    agent.update_risk_group()
                            
            self.epi_model()
            for agent in self.total_agent:
                agent.update_time()
                dic_list.append(agent.__dict__.copy())
        
        for t in trange(total_iteration - end_time):
            transition_matrices = {f'g{i}-{j}{k}': self.transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
            matrix[t+1+end_time] = transition_matrices
                
            self.decision_model(matrix[t+1 + end_time - lag])
            self.epi_model()

            for agent in self.total_agent:
                agent.update_time()
                dic_list.append(agent.__dict__.copy())

        return pd.DataFrame(dic_list)
    
    ### Run conditional or unconditional cash transfer policy
    def run_cash_transfer(self, start_time, end_time, total_iteration, cash_transfer, how, lag):

        self.inital_iteration = start_time
        matrix = {}
        transition_matrices_t0 = {f'g{i}-{j}{k}': self.init_transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
        matrix[0] = transition_matrices_t0
        
        self.decision_model(transition_matrices_t0)

        dic_list = []
        for agent in self.total_agent:
            dic_list.append(agent.__dict__.copy())

        for t in trange(self.inital_iteration):
            if t < lag:
                self.epi_model()
                transition_matrices = {f'g{i}-{j}{k}': self.transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
                matrix[t+1] = transition_matrices
                
                self.decision_model(transition_matrices_t0)

                for agent in self.total_agent:
                    agent.update_time()
                    dic_list.append(agent.__dict__.copy())
            
            else:
                self.epi_model()
                transition_matrices = {f'g{i}-{j}{k}': self.transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
                matrix[t+1] = transition_matrices
                
                self.decision_model(matrix[t+1 - lag])

                for agent in self.total_agent:
                    agent.update_time()
                    dic_list.append(agent.__dict__.copy())

        if how == 'unconditional':
            self.w_h += cash_transfer
            self.w_l += cash_transfer
            self.c_h += cash_transfer
            self.c_l += cash_transfer

        elif how == 'conditional':
            self.c_h += cash_transfer
            self.c_l += cash_transfer

        for t in trange(end_time - self.inital_iteration):
            self.epi_model()
            transition_matrices = {f'g{i}-{j}{k}': self.transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
            matrix[t+1 + self.inital_iteration] = transition_matrices
            self.decision_model(matrix[t+1 + self.inital_iteration - lag])

            for agent in self.total_agent:
                agent.update_time()
                dic_list.append(agent.__dict__.copy())
        
        if how == 'unconditional':
            self.w_h -= cash_transfer
            self.w_l -= cash_transfer
            self.c_h -= cash_transfer
            self.c_l -= cash_transfer
        elif how == 'conditional':
            self.c_h -= cash_transfer
            self.c_l -= cash_transfer
        
        for t in trange(total_iteration - end_time):
            self.epi_model()
            transition_matrices = {f'g{i}-{j}{k}': self.transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
            matrix[t+1 + end_time] = transition_matrices
            self.decision_model(matrix[t+1 + end_time - lag])

            for agent in self.total_agent:
                agent.update_time()
                dic_list.append(agent.__dict__.copy())

        return pd.DataFrame(dic_list)

    
    ### Paid sick leave policy

    def run_paid_sick_leave(self, start_time, end_time, total_iteration, pay_sick, lag):

        self.inital_iteration = start_time
        matrix = {}
        transition_matrices_t0 = {f'g{i}-{j}{k}': self.init_transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
        matrix[0] = transition_matrices_t0
        
        self.decision_model(transition_matrices_t0)

        dic_list = []
        for agent in self.total_agent:
            dic_list.append(agent.__dict__.copy())

        for t in trange(self.inital_iteration):
                if t < lag:
                    self.epi_model()
                    transition_matrices = {f'g{i}-{j}{k}': self.transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
                    matrix[t+1] = transition_matrices
            
                    self.decision_model(transition_matrices_t0)

                    for agent in self.total_agent:
                        agent.update_time()
                        dic_list.append(agent.__dict__.copy())
                else:
                    self.epi_model()
                    transition_matrices = {f'g{i}-{j}{k}': self.transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
                    matrix[t+1] = transition_matrices
                
                    self.decision_model(matrix[t+1 - lag])

                    for agent in self.total_agent:
                        agent.update_time()
                        dic_list.append(agent.__dict__.copy())
        
        self.pay_sick = pay_sick

        for t in trange(end_time - self.inital_iteration):
            self.epi_model()
            transition_matrices = {f'g{i}-{j}{k}': self.transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
            matrix[t+1 + self.inital_iteration] = transition_matrices
            self.decision_model(matrix[t+1 + self.inital_iteration - lag])

            for agent in self.total_agent:
                agent.update_time()
                dic_list.append(agent.__dict__.copy())
        
        self.pay_sick = 0

        for t in trange(total_iteration - end_time):
            self.epi_model()
            transition_matrices = {f'g{i}-{j}{k}': self.transition_matrix(f'g{i}-{j}{k}', 1) for i in range(2) for j in range(2) for k in range(2)}
            matrix[t+1 + end_time] = transition_matrices
            self.decision_model(matrix[t+1 + end_time - lag])
            for agent in self.total_agent:
                agent.update_time()
                dic_list.append(agent.__dict__.copy())

        return pd.DataFrame(dic_list)