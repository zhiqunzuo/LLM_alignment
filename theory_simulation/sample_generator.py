import numpy as np


class SampleGenerator():
    """
    sample generator
    """

    def __init__(self):
        pass

    def sample(self, N: int, seed: int) -> np.ndarray:
        raise NotImplementedError

    def getPercentile(self, alpha: float) -> float:
        raise NotImplementedError


class UniformSampleGenerator(SampleGenerator):
    """
    sample from uniform distribution
    """

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def sample(self, N: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        samples = rng.uniform(low=self.a, high=self.b, size=N)
        return samples

    def getPercentile(self, alpha: float) -> float:
        totalSize = 100000
        samples = self.sample(N=totalSize, seed=42)
        sortedSamples = np.sort(samples)
        return sortedSamples[int(totalSize * alpha - 1)]


class GaussianSampleGenerator(SampleGenerator):
    """
    sample from Gaussian distribution
    """

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def sample(self, N: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        samples = rng.normal(loc=self.mu, scale=self.sigma, size=N)
        return samples

    def getPercentile(self, alpha: float) -> float:
        totalSize = 100000
        samples = self.sample(N=totalSize, seed=42)
        sortedSamples = np.sort(samples)
        return sortedSamples[int(totalSize * alpha - 1)]
