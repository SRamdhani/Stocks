from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=False, unsafe_hash=True)
class envSetup:
    window: int = field(init=True, default=int, repr=False, compare=False)
    compData: int = field(init=True, default=int, repr=False, compare=False)
    rewardData: np.ndarray = field(init=True, default_factory=lambda: np.ndarray, repr=False, compare=False)
    actionDict: dict = field(init=True, default_factory=dict, repr=False, compare=False)
    balance: float = field(init=True, default=10000., repr=False, compare=False)
    reverseActionDict: dict = field(init=False, default=dict, repr=False, compare=False)
    maxiter: int = field(init=False, default=int, repr=False, compare=False)
    shares: int = field(init=False, default=0, repr=False, compare=False)
    sharePrice: int = field(init=False, default=0, repr=False, compare=False)
    origBalance: float = field(init=False, default=float, repr=False, compare=False)
    penalty: float = field(init=False, default=-100., repr=False, compare=False)
    predData: np.ndarray = field(init=False, default_factory=lambda: np.ndarray, repr=False, compare=False)

    def __post_init__(self):
        reverseActionDict = {self.actionDict[k]:k for k in self.actionDict}
        maxiter           = len(self.compData)-self.window
        origBalance       = self.balance
        predData          = self.compData[maxiter:(maxiter+self.window),:]

        object.__setattr__(self, 'reverseActionDict', reverseActionDict)
        object.__setattr__(self, 'maxiter', maxiter)
        object.__setattr__(self, 'origBalance', origBalance)
        object.__setattr__(self, 'predData', predData)

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