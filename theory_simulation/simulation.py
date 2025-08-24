import numpy as np

from sample_generator import SampleGenerator, UniformSampleGenerator, GaussianSampleGenerator
from typing import Tuple


def simulate(generator: SampleGenerator, batch: int, N: int, seed: int, alpha: float, delta: float) -> tuple[float, float, float]:
    numLargerThanPercentile = 0
    numMLM = 0
    numMLMLargerThanPercentile = 0
    for i in range(batch):
        samples = np.sort(generator.sample(N=N, seed=seed + i))
        mean = np.mean(samples[:-1])
        deviation = np.std(samples[:-1])
        bestSamples = samples[-1]

        percentile = generator.getPercentile(alpha=alpha)
        if bestSamples >= percentile:
            numLargerThanPercentile += 1

        if (bestSamples - mean) / deviation >= delta:
            numMLM += 1
            if bestSamples >= percentile:
                numMLMLargerThanPercentile += 1

    winRate = numLargerThanPercentile / batch
    winRateMLM = numMLMLargerThanPercentile / numMLM
    rateMLM = numMLM / batch
    return winRate, winRateMLM, rateMLM


def main():
    generator = UniformSampleGenerator(a=0, b=10)
    winRate, winRateMLM, rateMLM = simulate(
        generator=generator, batch=10000, N=5, seed=42, alpha=0.9, delta=1.8)
    print("winRate = {}, winRateMLM = {}, rateMLM = {}".format(
        winRate, winRateMLM, rateMLM))


if __name__ == "__main__":
    main()
