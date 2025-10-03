# Improving policy design and epidemic response using integrated models of economic choice and disease dynamics with behavioral feedback

![figure_1](https://github.com/user-attachments/assets/e6f25de4-f33d-48d7-aac9-2e8ada541194)

This repository provides the code for the Feedback-Informed Epidemiological Model (FIEM), which is introduced in the [pre-print](https://www.medrxiv.org/content/10.1101/2024.11.16.24317352v1):

Du, Zahn et al, "**Modeling dynamic disease-behavior feedbacks for improved epidemic prediction and response**", 2024 

To run the code, set up a virtual environment and install the dependencies:
# Clone this repository
git clone https://github.com/HopkinsIDD/epi-econ.git
cd epi-econ

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # (or `venv\Scripts\activate` on Windows)

# Install required packages
pip install -r requirements.txt
```
**/src**: Contains the main source code for the FIEM.
**/main.py**: Contains the code to run FIEM and base parameters.
- `results` – contains code to **replicate the figures** from the manuscript.  


## Parameters

Model parameters are controlled through an `OrderedDict` structure. Below is the full configuration:

```python
params = OrderedDict(
    initial_condition = [initial_space1], # Initial conditions
    n_epi = [4], # Number of epi updates between two decision updates
    beta = [0.2], # Infection rate
    r_base = [0.5], # Basis contact rate
    r1 = [1.5], # Additional contact on same health-economic status
    r2 = [4], # Additional contacts between two work risk groups
    r3 = [1.5], # Additional contacts between low-SES risk groups
    alpha = [0.0043], # Rate from recovered to susceptible
    gamma = [0.14], # Recovery rate
    theta_x = [-5], # Penalty for being sick
    theta_k = [3], # Extra sensitivity of vulnerable population
    theta_h = [0.5], # Sensitivity to hassle costs
    B = [0], # Payoff for making risky decision
    pc = [2], # Additional costs of risky behavior if infected
    mu = [0], # Mean of log hassle cost
    sigma = [0.25], # Std of log hassle cost
    threshold = [1e-5], # Tolerance of solving Bellman function
    v = [0.57721], # Euler’s constant
    k = [0.96], # Discount factor
    c_l = [15], # Baseline consumption for low-SES
    c_h = [66], # Baseline consumption for high-SES
    w_l = [98], # Wage for low-SES individuals
    w_h = [262], # Wage for high-SES individuals
    lag = [7],
    policy_name = ['conditional'], # Policy name, can choose from 'forced_behavior', 'unconditional', 'paid_sick_leave', and 'no_policy'
    # work_prob = [0.7], # Proportion of work after policy
    policy_start = [20], # Start time of policy
    policy_end = [140], # End time of policy
    total_time = [140], # Total time of simulation
    cash_transfer = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80], # Cash transfer amount
    pay_sick = [90], # Sick-pay transfer amount
    how = ['random'] # How to assign people to not work under labor restriction
)
