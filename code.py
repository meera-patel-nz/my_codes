import numpy as np
import matplotlib.pyplot as plt

# true values
m_true=2.0
b_true=0.5
x=np.arange(0, 25, 0.5) # Note: because this is a numpy array, we don't need a loop, because the formula below is applied to every element at once.
y_true=(m_true*x)+b_true

# generate noise and add it to the data
sigma=1.0
noise=np.random.normal(loc=0.0, scale=sigma, size=len(x)) 
# loc,  the mean of the noise (centred on 0)
# scale, the standard deviation (how wide the scatter is)
# size, len(x) makes it return an array of random numbers

y_obs=y_true + noise
plt.plot(x,y_obs)
plt.title("Fake noisy line data")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# initial guesses for m and b. (starting far from true values deliberately)
m_current=5.0
b_current=2.0

# initial step sizes (should they be different??)
step_m=0.03
step_b=0.4

n_steps=100000

# arrays to track the chain

'''
Explaining np.zeros:
- reserving a row of empty boxes before you know what is going in them.
- Why make empty boxes? Because when the MCMC loop runs, at each step we want to write a value into the 'next' box.
- pre-allocating the zeroes, rather than starting with an empty array makes it faster, as numpy knows the size of the array instead of having to resize it for each step.
'''

m_chain=np.zeros(n_steps)
b_chain=np.zeros(n_steps)
chi2_chain=np.zeros(n_steps)
# make an aray that I can append onto, because the acceptance will be different for m and b , as well as the step size. 
# 2 separate growing lists, for whenver an m or b value is accepted
accept_m=[]
accept_b=[]
param_varied=np.zeros(n_steps)

#chi squared function (the 'goodness' of the fit)
# small chi squared: line passes close to data points (good fit)
# large chi squared: line is far from data points (bad fit)
def chi_squared(m, b, x, y_obs, sigma): # m and b are parameters here, they don't need to exist elsewhere
    y_model = (m * x) + b
    chi2 = np.sum((y_obs - y_model)**2 / sigma**2)
    return chi2

chi2_current = chi_squared(m_current, b_current, x, y_obs, sigma)

'''
What are we looping an why:
- for each 'flip of the coin' either m or b is going to be varied. (if , else statement)
- param_varied[i] picks and records which parameter this particular step will vary
- m_proposed etc. generates a random number centred on where the m
'''

for i in range(n_steps):
    coin = np.random.uniform(0, 1) # draws one random decimal between 0 and 1
    
    if coin < 0.5:
        param_varied[i] = 0 # m is varied
        m_proposed = np.random.normal(loc=m_current, scale=step_m) # draws one random decimal from a normal distribution centred on m_current, with a width of step_m
        b_proposed = b_current # b stays the same
    else:
        param_varied[i] = 1 # b is varied
        m_proposed = m_current # m stays the same
        b_proposed = np.random.normal(loc=b_current, scale=step_b) # draws one random decimal from a normal distribution centred on b_current, with a width of step_b
      
    # calculate chi squared for proposed step   
    chi2_proposed = chi_squared(m_proposed, b_proposed, x, y_obs, sigma)
    
    # decide whether to accept or reject
    delta_chi2 = chi2_proposed - chi2_current

    if delta_chi2 < 0: # if the proposed step is better (smaller chi squared), accept it
        accept = True

    else: # if the proposed step is worse (larger chi squared), accept it with a probability of exp(-delta_chi2/2)
        prob_ratio = np.exp(-delta_chi2 / 2)
        random_number = np.random.uniform(0, 1)
        if prob_ratio > random_number:
            accept = True
        else:
            accept = False
    
    # update current values if accepted, and record acceptance. Note, else means nothing happens .
    if accept:
        m_current = m_proposed
        b_current = b_proposed
        chi2_current = chi2_proposed
    
    if param_varied[i] == 0: # the double == means 'check if these are equal' --> (yes or no question, giving back true or false, but doesn't change anything)
        if accept:
            accept_m.append(1)  # m-step, accepted → record a 1
        else:
            accept_m.append(0)  # m-step, rejected → record a 0
    else:                       # this was a b-step
        if accept:
            accept_b.append(1) # b-step, accepted → record a 1
        else:
            accept_b.append(0)  # b-step, rejected → record a 0

    # record the chain, every step, whether accepted or not
    m_chain[i] = m_current
    b_chain[i] = b_current
    chi2_chain[i] = chi2_current            
     
#check:
print("m acceptance fraction:", np.mean(accept_m))
print("b acceptance fraction:", np.mean(accept_b))

# The target for each is somewhere between 0.2 and 0.5, if either is way outside that range, adjust step_m or step_b
# too low acceptance --> Step is too big (shrink step size)
# high acceptance --> Step is too small, grow it)
     
'''
Determine the burn-in period!

plt.plot(chi2_chain[:2000]) # only the first 2000 steps
plt.xlabel("step")
plt.ylabel("chi^2")
plt.title("First 2000 steps")
plt.show()
'''
# watch  the wandering during burn-in for m and b, and confirm visually where it settles down.

plt.clf()
plt.plot(m_chain[:2000])
plt.xlabel("step")
plt.ylabel("m")
plt.title("m vs step (full chain, including burn-in)")
plt.show()

plt.clf()
plt.plot(b_chain[:2000])
plt.xlabel("step")
plt.ylabel("b")
plt.title("b vs step (full chain, including burn-in)")
plt.show()

burn_in = 1000
m_chain_clean = m_chain[burn_in:] # this syntax essentially means , leave everything before burn_in, and go all the way to the end.
b_chain_clean = b_chain[burn_in:]
chi2_chain_clean = chi2_chain[burn_in:]

'''
plt.plot(chi2_chain_clean)
plt.xlabel("step (post burn-in)")
plt.ylabel("chi^2")
plt.title("After removing burn-in")
plt.show()
'''