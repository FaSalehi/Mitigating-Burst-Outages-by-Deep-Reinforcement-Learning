"""Define the power and blocklength control environment by generating signals 
to mitigate the consecutive errors"""

import gymnasium as gym
import numpy as np
from scipy import special as sp
from scipy.signal import lfilter

def qfunc(x):
    return 0.5 - 0.5 * sp.erf(x / np.sqrt(2))

def qfuncinv(x):
    return np.sqrt(2) * sp.erfinv(1 - 2*x)

class PowerControlEnv(gym.Env):
    def __init__(
        self,
        max_blocklength,
        max_snr_dB,
        tgtOutageProb,
        packetLength,
        consErrorThr,
        errorRateThr,
        w1, w2,
        max_time = 100, #1000,  # Max number of steps per episode
        seed = None    
    ):
        
        self.max_blocklength = max_blocklength
        self.max_snr_dB = max_snr_dB
        self.tgtOutageProb = tgtOutageProb
        self.packetLength = packetLength
        self.consErrorThr = consErrorThr
        self.errorRateThr = errorRateThr
        self.max_time = max_time
        self.seed_value = seed
        self.w1 = w1
        self.w2 = w2
        self.counter = 0
        self.time_slot = 0
        self.consec_error = 0

        ### Generate the interference signal
        nrOfRuns = int(1e7)  #number of interference samples
        nrOfInterferers = 5  #10 
        transDuration = 10
        activProb = 1  #0.5
        inr_db = np.array([5, 2, 0, -3, -10]) # mean INRs of the interferers in dB
        inr_lin = 10 ** (inr_db / 10)  # mean INRs of the interferers

        filterLength = 100  # to make correlation between samples
        filterCoefs = 1/filterLength * np.ones(filterLength)
        filterStateI = np.zeros([filterLength - 1, nrOfInterferers])
        filterStateQ = np.zeros([filterLength - 1, nrOfInterferers])

        self.sumInterference = 0
        for ii in range(nrOfInterferers):
            actInterferers = np.zeros(nrOfRuns)
            num = 1
            while num <= nrOfRuns:
                var = np.random.rand()
                if var <= activProb:
                    actInterferers[num-1:min(num-1 + transDuration, nrOfRuns)] = 1
                    num += transDuration
                else:
                    actInterferers[num-1] = 0
                    num += 1

            noiseI = np.random.randn(nrOfRuns) * actInterferers
            noiseQ = np.random.randn(nrOfRuns) * actInterferers
            
            noiseI, stateI = lfilter(filterCoefs, 1, noiseI.flatten(), zi=filterStateI[:, ii])
            noiseQ, stateQ = lfilter(filterCoefs, 1, noiseQ.flatten(), zi=filterStateQ[:, ii])
            filterStateI[:, ii] = stateI
            filterStateQ[:, ii] = stateQ
            
            nn = noiseI + 1j * noiseQ
            interfChannel = np.real(nn)**2 + np.imag(nn)**2
            interfSignal = interfChannel / np.mean(interfChannel) * inr_lin[ii]
            
            self.sumInterference += interfSignal

        ### Generate the desired channel
        noiseI = np.random.randn(nrOfRuns)
        noiseQ = np.random.randn(nrOfRuns)

        noiseI, stateI = lfilter(filterCoefs, 1, noiseI.flatten(), zi=filterStateI[:, ii])
        noiseQ, stateQ = lfilter(filterCoefs, 1, noiseQ.flatten(), zi=filterStateQ[:, ii])
        filterStateI[:, ii] = stateI
        filterStateQ[:, ii] = stateQ

        nn = noiseI + 1j * noiseQ
        desiredChannel = np.real(nn)**2 + np.imag(nn)**2
        self.channel_gains = (desiredChannel / np.mean(desiredChannel))
    
        ### Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  #Continuous power level and blocklength
        self.observation_space = gym.spaces.Box(low=0, high=50, shape=(1,), dtype=np.float64)  #Continuous SINR levels: |h|^2/(n^2+I^2)
        
        # Calculation for reward (EE) normalization
        blocklength_range = np.array([1,self.max_blocklength+1])
        power_range_dB = np.array([-self.max_snr_dB, self.max_snr_dB])
        power_range = 10 ** (power_range_dB/10) 
        bl_power = blocklength_range * power_range
        energy_efficiency = self.packetLength/bl_power
        self.EE_min = np.min(energy_efficiency)
        self.EE_max = np.max(energy_efficiency)

    def step(self, action):

        # Update user power and blocklength level based on action
        power_level = 10 ** (self.max_snr_dB * action[0]/10)
        mean_blocklength = (1 + self.max_blocklength)/2
        blocklength = action[1] * (self.max_blocklength - mean_blocklength) + mean_blocklength
        blocklength = np.min([np.ceil(blocklength).item(), self.max_blocklength])

        # Calculate the outage probability
        actual_sinr = power_level * self.channel_gains[self.counter] / (1 + self.sumInterference[self.counter])
        C_actualSinr = np.log2(1 + actual_sinr)
        V_actualSinr = 1/(np.log(2)**2) * (1 - 1/((1 + actual_sinr)**2))
        outageProb = qfunc((blocklength * C_actualSinr - self.packetLength)/np.sqrt(blocklength * V_actualSinr))

        # Return the user's actual SINR
        actual_sinr_normalized = self.channel_gains[self.counter] / (1 + self.sumInterference[self.counter])
        new_state = np.array([actual_sinr_normalized])

        # Calculate the consumed energy
        consumed_energy = blocklength * power_level

        # Calculate the user energy efficiency (reward)
        reward_EE = self.packetLength/consumed_energy
        #reward_energy = 1 - consumed_energy/(self.max_power*self.max_blocklength)

        # Normalize the reward (max-min normalization)
        reward_EE = (reward_EE - self.EE_min)/(self.EE_max - self.EE_min)

        # Accumulate the consecutive errors and define a penalty for outage
        if outageProb > self.tgtOutageProb:
            self.consec_error += 1
            penalty_outage = -1.0
        else:
            self.consec_error = 0
            penalty_outage = 0.0

        # Return the current state index (number of consecutive errors)
        #new_state = np.min([self.consec_error, self.consErrorThr + 1])

        # Define a penalty for consecutive errors
        if self.consec_error > self.consErrorThr:
          penalty_CE = -1.0  #-10.0
        else:
          penalty_CE = 0.0

        # The total reward is the sum of the rewards elements
        self.w3 = 1.0 - self.w1 - self.w2
        reward = self.w1 * reward_EE + self.w2 * penalty_outage + self.w3 * penalty_CE

        # Return the system's information
        info = {
            "blocklength":blocklength,
            "consumed_energy":consumed_energy,
            "outageProb":outageProb,
            "reward_EE":reward_EE,
            "penalty_outage":penalty_outage,
            "penalty_CE":penalty_CE,
            "reward_weights":[self.w1, self.w2, self.w3]
            }

        self.counter += 1
        self.time_slot = self.time_slot + 1

        # Determine if the episode is done (e.g., when a certain number of steps are reached)
        terminated = self._is_terminated()
        # Termination due to timeout (max number of steps per episode)
        truncated = self._is_truncated()

        return new_state, reward.item(), terminated, truncated, info
    
    def _is_terminated(self):
        if self.counter >= len(self.channel_gains)-1:
            return True
        return False
    
    def _is_truncated(self):
        if self.time_slot >= self.max_time:
            return True
        return False
    
    def reset(self, seed=None):
        # Reset the environment to an initial state
        self.time_slot = 0
        if seed is not None:
            self.seed_value = seed
        initial_state = self.observation_space.sample()
        new_state = initial_state

        return new_state, {}
    

if __name__ == "__main__":
    # It will check your custom environment and output additional warnings if needed
    from stable_baselines3.common.env_checker import check_env
    env = PowerControlEnv(
                    max_blocklength=int(1e3),
                    max_snr_dB=20.0,
                    tgtOutageProb=1e-5,
                    packetLength=50,
                    consErrorThr=5,
                    errorRateThr=0.01,
                    w1 = 0.7,
                    w2 = 0.3,
                    )
    check_env(env)