from dataclasses import dataclass

@dataclass(frozen=False, unsafe_hash=True)
class envSetup:
    def __init__(self, window, compData, rewardData,
                 actionDict, balance = 10000):
        self.window            = window
        self.compData          = compData
        self.rewardData        = rewardData
        self.actionDict        = actionDict
        self.reverseActionDict = {actionDict[k]:k for k in actionDict}
        self.balance           = balance
        self.maxiter           = len(compData)-window
        self.shares            = 0
        self.sharePrice        = 0
        self.origBalance       = balance
        self.penalty           = -100
        self.predData          = compData[self.maxiter:(self.maxiter+window),:]

    def step(self, it, action):
        # Check Done status and get New Observation.
        done = False
        compData = self.compData[(it):(it+self.window)]
        reward = 0
        # Get price to base things off of.
        sharePrice = self.rewardData[(it+self.window), -1]
        self.sharePrice = sharePrice
        # Do the action.
        getAction = self.actionDict[action]
        if getAction == 'buy':
            if self.shares == 0:
                self.shares = self.balance/sharePrice
                reward = -1*self.balance
                self.balance = 0
            else:
                reward = self.penalty
        elif getAction == 'sell':
            if self.shares > 0:
                self.balance = sharePrice * self.shares
                self.shares = 0
                reward = self.balance
            else:
                reward = self.penalty
        if it==(self.maxiter-1):
            done=True
        return compData.flatten(), reward, done, None

    def reset(self):
        self.balance = self.origBalance
        self.shares = 0
        compData = self.compData[(0):(0+self.window)]
        # sharePrice = self.rewardData[self.window, -1]
        return compData.flatten()