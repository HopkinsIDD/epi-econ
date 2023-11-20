import pandas as pd
import numpy as np
from tqdm import trange
import random

risk_group_mapping = pd.DataFrame(columns=['Risk_group', 'Decision', 'State_variable_VUL', 'State_variable_SES'])
risk_group_mapping['Risk_group'] = ['g0-00', 'g0-01', 'g0-10', 'g0-11', 'g1-00', 'g1-01', 'g1-10', 'g1-11']
risk_group_mapping['Decision'] = ['not-work', 'not-work', 'not-work', 'not-work',
                                  'work', 'work', 'work', 'work']
risk_group_mapping['State_variable_VUL'] = ['not-vulnerable', 'not-vulnerable', 'vulnerable', 'vulnerable',
                                            'not-vulnerable', 'not-vulnerable', 'vulnerable', 'vulnerable']
risk_group_mapping['State_variable_SES'] = ['high-SES', 'low-SES', 'high-SES', 'low-SES',
                                            'high-SES', 'low-SES', 'high-SES', 'low-SES']


# Create each individual
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
    """

    def __init__(self, m, t, x, g):
        self.index = m
        self.time = t
        self.infection_status = x
        self.risk_group = g

        if self.risk_group == 'g0-00':
            self.decision = 'not-work'
            self.state_VUL = 'not-vulnerable'
            self.state_SES = 'high-SES'
        elif self.risk_group == 'g0-01':
            self.decision = 'not-work'
            self.state_VUL = 'not-vulnerable'
            self.state_SES = 'low-SES'
        elif self.risk_group == 'g0-10':
            self.decision = 'not-work'
            self.state_VUL = 'vulnerable'
            self.state_SES = 'high-SES'
        elif self.risk_group == 'g0-11':
            self.decision = 'not-work'
            self.state_VUL = 'vulnerable'
            self.state_SES = 'low-SES'

        elif self.risk_group == 'g1-00':
            self.decision = 'work'
            self.state_VUL = 'not-vulnerable'
            self.state_SES = 'high-SES'
        elif self.risk_group == 'g1-01':
            self.decision = 'work'
            self.state_VUL = 'not-vulnerable'
            self.state_SES = 'low-SES'
        elif self.risk_group == 'g1-10':
            self.decision = 'work'
            self.state_VUL = 'vulnerable'
            self.state_SES = 'high-SES'
        elif self.risk_group == 'g1-11':
            self.decision = 'work'
            self.state_VUL = 'vulnerable'
            self.state_SES = 'low-SES'

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


# Create model with decision feedback
class Model:
    def __init__(self, initial_condition, n_epi, beta, r_base, r1, r2, r3,
                 alpha, gamma, theta_x, theta_k, theta_h, B, pc,
                 mu=0, sigma=0.5, threshold=1e-5, v=0.57721, k=0.9,
                 c_l=13, c_h=29, w_l=51, w_h=115):
        self.iteration = None
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
                if row == col:
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

    def transition_matrix(self, risk_group, n):
        p_si = 0
        N = len(self.total_agent)
        for g in {agent.risk_group for agent in self.total_agent}:
            I_g = sum((agent.risk_group == g) & (agent.infection_status == "i") for agent in self.total_agent)
            # N_g = sum(agent.risk_group == g for agent in self.total_agent)
            # N_g = len(self.total_agent)
            p_si += (I_g * self.Beta.loc[risk_group, g] / n) / N
        if p_si >= 1:
            p_si = 1

        p_ss = 1 - float(repr(p_si))

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

    def utility_func(self, infection_status, state_VUL, state_SES, decision, hassle_cost_group):

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

        if x == 2:
            f = (1 + self.theta_k * k) * self.theta_x * x
        else:
            f = self.theta_x * x

        if decision == 'not-work':
            U = f + np.log(c_base)
        elif (decision == 'work') & (infection_status == 'i'):
            U = f + np.log(w) + self.B - (1 + self.pc) * self.theta_h * h
        else:
            U = f + np.log(w) + self.B - self.theta_h * h

        return np.float64(U).item()

    def bellman_func(self, state_VUL, state_SES, hassle_cost_group,
                     p_g0_00, p_g0_01, p_g0_10, p_g0_11,
                     p_g1_00, p_g1_01, p_g1_10, p_g1_11):

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
            if j > 1000:
                break

        p_decision = pd.DataFrame((np.exp(V) / np.exp(V).sum(axis=1)[:, None]),
                                  index=['s', 'i', 'r'],
                                  columns=['work', 'not-work'])

        return p_decision

    def run(self, iteration):

        self.iteration = iteration

        dic_list = []
        for agent in self.total_agent:
            dic = {
                'm': agent.index,
                't': agent.time,
                'decision': agent.decision,
                'state_VUL': agent.state_VUL,
                'state_SES': agent.state_SES,
                'infection_status': agent.infection_status,
                'risk_group': agent.risk_group,
                'hassle_cost_group': agent.hassle_cost_group
            }
            dic_list.append(dic)

        for i in trange(self.iteration):
            # pre calculate transition proabbility
            p_g0_00 = self.transition_matrix('g0-00', 1)
            p_g0_01 = self.transition_matrix('g0-01', 1)
            p_g0_10 = self.transition_matrix('g0-10', 1)
            p_g0_11 = self.transition_matrix('g0-11', 1)
            p_g1_00 = self.transition_matrix('g1-00', 1)
            p_g1_01 = self.transition_matrix('g1-01', 1)
            p_g1_10 = self.transition_matrix('g1-10', 1)
            p_g1_11 = self.transition_matrix('g1-11', 1)

            DDPs_list = {}
            for state_VUL in risk_group_mapping['State_variable_VUL'].unique():
                for state_SES in risk_group_mapping['State_variable_SES'].unique():
                    for hassle_cost_group in ['low', 'median', 'high']:
                        DDPs_list[state_VUL, state_SES, hassle_cost_group] = \
                            self.bellman_func(state_VUL, state_SES, hassle_cost_group,
                                              p_g0_00, p_g0_01, p_g0_10, p_g0_11,
                                              p_g1_00, p_g1_01, p_g1_10, p_g1_11)

            for agent in self.total_agent:
                p_decision = DDPs_list[agent.state_VUL, agent.state_SES, agent.hassle_cost_group]
                agent.update_decision(p_decision)
                agent.update_time()

            # update transition matrix
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
                        # rg_S[i].time += 1
                    for j in indices_I_to_R:
                        rg_I[j].infection_status = 'r'
                        # S[i].time += 1
                    for j in indices_R_to_S:
                        rg_R[j].infection_status = 's'

                if t == self.n_epi - 1:
                    for agent in self.total_agent:
                        # agent.update_time()
                        dic = {
                            'm': agent.index,
                            't': agent.time,
                            'decision': agent.decision,
                            'state_VUL': agent.state_VUL,
                            'state_SES': agent.state_SES,
                            'infection_status': agent.infection_status,
                            'risk_group': agent.risk_group,
                            'hassle_cost_group': agent.hassle_cost_group
                        }
                        dic_list.append(dic)

        df = pd.DataFrame(dic_list)

        return df

    def run_basic_SIR(self, iteration):

        self.iteration = iteration
        dic_list = []
        for agent in self.total_agent:
            dic = {
                    'm': agent.index,
                    't': agent.time,
                    'decision': agent.decision,
                    'state_VUL': agent.state_VUL,
                    'state_SES': agent.state_SES,
                    'infection_status': agent.infection_status,
                    'risk_group': agent.risk_group,
                    'hassle_cost_group': agent.hassle_cost_group
                }
            dic_list.append(dic)

        for i in trange(self.iteration):
            for t in range(self.n_epi):

                transition_matrix_list = {}
                for rg in risk_group_mapping['Risk_group'].unique():
                    transition_matrix_list[rg] = self.transition_matrix(rg, self.n_epi)

                '''
                Update infection
                '''
                for rg in risk_group_mapping['Risk_group'].unique():
                    rg_S = [agent for agent in self.total_agent if agent.risk_group == rg\
                              and agent.infection_status == 's']
                    rg_I = [agent for agent in self.total_agent if agent.risk_group == rg\
                         and agent.infection_status == 'i']
                    rg_R = [agent for agent in self.total_agent if agent.risk_group == rg\
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
                        # rg_S[i].time += 1
                    for j in indices_I_to_R:
                        rg_I[j].infection_status = 'r'
                        # S[i].time += 1
                    for j in indices_R_to_S:
                        rg_R[j].infection_status = 's'

                if t == self.n_epi - 1:
                    for agent in self.total_agent:
                        agent.update_time()
                        dic = {
                                'm': agent.index,
                                't': agent.time,
                                'decision': agent.decision,
                                'state_VUL': agent.state_VUL,
                                'state_SES': agent.state_SES,
                                'infection_status': agent.infection_status,
                                'risk_group': agent.risk_group,
                                'hassle_cost_group': agent.hassle_cost_group
                            }
                        dic_list.append(dic)

        df = pd.DataFrame(dic_list)

        return df
