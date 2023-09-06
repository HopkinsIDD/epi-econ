import pandas as pd
import numpy as np
from tqdm import trange
import random
risk_group_mapping = pd.DataFrame(columns = ['Risk_group', 'Decision', 'State_variable'])
risk_group_mapping['Risk_group'] = ['g1', 'g2', 'g3', 'g4']
risk_group_mapping['Decision'] = ['risky', 'non-risky', 'risky', 'non-risky']
risk_group_mapping['State_variable'] = ['vulnerable', 'vulnerable', 'non-vulnerable', 'non-vulnerable']


###### Create each individual

class Agent:
    '''
    Create each individual with different attributes

    Attributes:
        index: the index for each individual
        time: runing time
        deicision: risky or non-risk
        state: state variable
        infection_status: s, i, r
        risk_group: depends on state and infection_status
        hassle_cost_group: three discreted hassle cost group
    '''

    def __init__(self, m, t, x, g):
        self.index = m
        self.time = t
        self.infection_status = x
        self.risk_group = g

        if self.risk_group == 'g1':
            self.decision = 'risky'
            self.state = 'vulnerable'
        elif self.risk_group == 'g2':
            self.decision = 'non-risky'
            self.state = 'vulnerable'
        elif self.risk_group == 'g3':
            self.decision = 'risky'
            self.state = 'non-vulnerable'
        elif self.risk_group == 'g4':
            self.decision = 'non-risky'
            self.state = 'non-vulnerable'

        self.hassle_cost_group = 'low'

    def __repr__(self):
        '''String representation of Agent'''
        return 'Agent with index = {}'.format(self.index) + \
            ', time = {}'.format(self.time) + \
            ', risk group = {}'.format(self.risk_group) + \
            ', infection status = {}'.format(self.infection_status)

    def update_time(self):
        '''update running time'''
        self.time = self.time + 1

    def update_decision(self, p_decision):
        '''update decision based on output from Bellman function
        ----------
        Parameters
        p_decision: probability decision matrix

        ----------
        Outputs
        Updtate decision and risk group
        '''
        if self.infection_status == 's':
            p = p_decision.iloc[0, 0]
        elif self.infection_status == 'i':
            p = p_decision.iloc[1, 0]
        elif self.infection_status == 'r':
            p = p_decision.iloc[2, 0]
        rv = np.random.uniform(0, 1, 1).item()
        if rv < p:
            self.decision = 'risky'
        else:
            self.decision = 'non-risky'

        if self.state == 'vulnerable':
            if self.decision == 'risky':
                self.risk_group = 'g1'
            else:
                self.risk_group = 'g2'

        if self.state == 'non-vulnerable':
            if self.decision == 'risky':
                self.risk_group = 'g3'
            else:
                self.risk_group = 'g4'

    def update_infection_status(self, p):
        '''update infection status based on transition matrix
        ----------
        Parameters
        p: transition probability matrix 3x3
        ----------
        Output
        update infection status at individual level
        '''
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


### Create model with decision feedback
### Create model with decision feedback
class Model:
    def __init__(self, initial_condition, n_epi, beta, C_ij, phi_i,
                 alpha, gamma, theta_x, theta_k, theta_h, B, pc,
                 mu=0, sigma=0.5, threshold=1e-5, v=0.57721, k=0.9):
        '''
        Create simulation model
        ----------
        Parameters
        initial_condition: Set up initial population.
        n_epi: Number of epi simulation within a day.
        beta: Intrinsic transmission rate of infection.
        C_ij: Contact matrix.
        phi_i: Relative susceptibility of individuals in group i.
        alpha: Rate of waning of risk group g.
        gamma: Rate of recovery of risk group g.
        theta_x: Penalty for being sick.
        theta_k: Extra sensitivity of vulnerable population.
        theta_h: Sensitivity to hassle costs.
        pc: Additional costs of risky behavior if infected.
        mu=0, sigma=0.5: mean and var of hassle cost
        threshold: error upper bound for solving Bellman equation.
        v: Euler’s constant.
        k: Discounting factor.
        ----------
        '''
        self.total_agent = []
        for spec in initial_condition:
            for i in range(spec['amount']):
                new_agent = Agent(len(self.total_agent), 0, spec['infection_status'],
                                  spec['risk_group'])
                self.total_agent.append(new_agent)

        # create random index for hassle cost group
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

        self.beta = pd.DataFrame(np.array(beta * C_ij * phi_i),
                                 index=['g1', 'g2', 'g3', 'g4'],
                                 columns=['g1', 'g2', 'g3', 'g4'])

        self.alpha = pd.DataFrame(np.array([alpha, alpha, alpha, alpha]),
                                  index=['g1', 'g2', 'g3', 'g4'],
                                  columns=['alpha'])

        self.gamma = pd.DataFrame(np.array([gamma, gamma, gamma, gamma]),
                                  index=['g1', 'g2', 'g3', 'g4'],
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

    def transition_matrix(self, risk_group, n):
        '''
        :param risk_group: string, e.g. 'g1'
        :param n: number of epi updates within a day
        :return: 3*3 transition matrix
        '''
        p_si = 0
        for g in {agent.risk_group for agent in self.total_agent}:
            I_g = sum((agent.risk_group == g) & (agent.infection_status == "i") for agent in self.total_agent)
            N_g = sum(agent.risk_group == g for agent in self.total_agent)
            p_si += (I_g * self.beta.loc[risk_group, g] / n) / N_g
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
                         index=['s(t)', 'i(t)', 'r(t)'],
                         columns=['s(t+1)', 'i(t+1)', 'r(t+1)'])
        return p

    def utility_func(self, infection_status, state, decision, hassle_cost_group):
        '''
        :param infection_status: string {'s', 'i', 'r'}
        :param state: string {'vulnerable', 'non-vulnerable'}
        :param decision: string {'risky', 'non-risky'}
        :param hassle_cost_group: string {'high', 'median', 'low'}
        :return: float value of utility
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

        if state == 'vulnerable':
            k = 1
        else:
            k = 0

        if x == 2:
            f = (1 + self.theta_k * k) * self.theta_x * x
        else:
            f = self.theta_x * x

        if decision == 'non-risky':
            U = f
        elif (decision == 'risky') & (infection_status == 'i'):
            U = f + self.B - (1 + self.pc) * self.theta_h * h
        else:
            U = f + self.B - self.theta_h * h

        return np.float64(U).item()

    def bellman_func(self, state, hassle_cost_group, p_g1, p_g2, p_g3, p_g4):
        '''
        :param state: string {'vulnerable', 'non-vulnerable'}
        :param hassle_cost_group: string {'high', 'median', 'low'}
        :param p_g1: 3*3 transition matrix g1
        :param p_g2: 3*3 transition matrix g2
        :param p_g3: 3*3 transition matrix g3
        :param p_g4: 3*3 transition matrix g4
        :return: decisions probability matrix
        '''

        V = np.ones((3, 2))
        U = np.zeros((3, 2))
        V_new = np.ones((3, 2))

        error = 1
        i = 0
        for decision in ['risky', 'non-risky']:
            u = []
            for infection_status in ['s', 'i', 'r']:
                u.append(self.utility_func(infection_status, state, decision, hassle_cost_group))
            U[:, i] = u
            i += 1

        if state == 'vulnerable':
            p1 = p_g1
            p2 = p_g2
        if state == 'non-vulnerable':
            p1 = p_g3
            p2 = p_g4

        j = 0
        while error > self.threshold:
            V_new[:, 0] = U[:, 0] + self.k * p1.to_numpy().dot((np.log(np.exp(V).sum(axis=1)) + self.v))
            V_new[:, 1] = U[:, 1] + self.k * p2.to_numpy().dot((np.log(np.exp(V).sum(axis=1)) + self.v))
            error = abs(V - V_new).sum()
            V = V_new.copy()
            j += 1
            if j > 1000:
                break

        p_decision = pd.DataFrame((np.exp(V) / np.exp(V).sum(axis=1)[:, None]),
                                  index=['s', 'i', 'r'],
                                  columns=['risky', 'non-risky'])

        return p_decision

    def run(self, iteration):

        self.iteration = iteration

        dic_list = []
        for agent in self.total_agent:
            dic = {
                'm': agent.index,
                't': agent.time,
                'decision': agent.decision,
                'state_variable': agent.state,
                'infection_status': agent.infection_status,
                'risk_group': agent.risk_group,
                'hassle_cost_group': agent.hassle_cost_group
            }
            dic_list.append(dic)

        for i in trange(self.iteration):
            # pre calculate transition proabbility
            p_g1 = self.transition_matrix('g1', 1)
            p_g2 = self.transition_matrix('g2', 1)
            p_g3 = self.transition_matrix('g3', 1)
            p_g4 = self.transition_matrix('g4', 1)
            #             print(p_g1)

            DDPs_list = {}
            for state in risk_group_mapping['State_variable'].unique():
                for hassle_cost_group in ['low', 'median', 'high']:
                    DDPs_list[state, hassle_cost_group] = \
                        self.bellman_func(state, hassle_cost_group, p_g1, p_g2, p_g3, p_g4)

            for agent in self.total_agent:
                p_decision = DDPs_list[agent.state, agent.hassle_cost_group]
                agent.update_decision(p_decision)
                agent.update_time()

            # update transition matrix
            for j in range(self.n_epi):

                p_g1 = self.transition_matrix('g1', self.n_epi)
                p_g2 = self.transition_matrix('g2', self.n_epi)
                p_g3 = self.transition_matrix('g3', self.n_epi)
                p_g4 = self.transition_matrix('g4', self.n_epi)

                for agent in self.total_agent:
                    if agent.risk_group == 'g1':
                        agent.update_infection_status(p_g1)

                    elif agent.risk_group == 'g2':
                        agent.update_infection_status(p_g2)

                    elif agent.risk_group == 'g3':
                        agent.update_infection_status(p_g3)

                    elif agent.risk_group == 'g4':
                        agent.update_infection_status(p_g4)

                    if j == self.n_epi - 1:
                        dic = {
                            'm': agent.index,
                            't': agent.time,
                            'decision': agent.decision,
                            'state_variable': agent.state,
                            'infection_status': agent.infection_status,
                            'risk_group': agent.risk_group,
                            'hassle_cost_group': agent.hassle_cost_group
                        }
                        dic_list.append(dic)
        df = pd.DataFrame(dic_list)

        return df

    def run_basic_SIR(self, iteration):
        '''
        SIR model without decision feedback
        '''
        self.iteration = iteration

        dic_list = []
        for agent in self.total_agent:
            dic = {
                'm': agent.index,
                't': agent.time,
                'decision': agent.decision,
                'state_variable': agent.state,
                'infection_status': agent.infection_status,
                'risk_group': agent.risk_group
            }
            dic_list.append(dic)

        for i in trange(self.iteration):
            for j in range(self.n_epi):
                p_g1 = self.transition_matrix('g1', self.n_epi)
                p_g2 = self.transition_matrix('g2', self.n_epi)
                p_g3 = self.transition_matrix('g3', self.n_epi)
                p_g4 = self.transition_matrix('g4', self.n_epi)

                for agent in self.total_agent:
                    if agent.risk_group == 'g1':
                        agent.update_infection_status(p_g1)

                    elif agent.risk_group == 'g2':
                        agent.update_infection_status(p_g2)

                    elif agent.risk_group == 'g3':
                        agent.update_infection_status(p_g3)

                    elif agent.risk_group == 'g4':
                        agent.update_infection_status(p_g4)

                    if j == self.n_epi - 1:
                        agent.update_time()
                        dic = {
                            'm': agent.index,
                            't': agent.time,
                            'decision': agent.decision,
                            'state_variable': agent.state,
                            'infection_status': agent.infection_status,
                            'risk_group': agent.risk_group,
                            'hassle_cost_group': agent.hassle_cost_group
                        }
                        dic_list.append(dic)

        df = pd.DataFrame(dic_list)

        return df

