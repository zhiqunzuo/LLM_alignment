import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from sample_generator import SampleGenerator, UniformSampleGenerator, GaussianSampleGenerator
from typing import Tuple


def quantile_ci(data, p=0.5, alpha=0.05):
    """
    Nonparametric confidence interval for the p-th quantile
    using order statistics (binomial distribution).
    """
    data = np.sort(np.asarray(data))
    n = len(data)

    lower_idx = st.binom.ppf(alpha/2, n, p)
    upper_idx = st.binom.isf(alpha/2, n, p)

    lower_idx = int(max(0, np.floor(lower_idx)))
    upper_idx = int(min(n-1, np.ceil(upper_idx)))

    qhat = np.quantile(data, p)

    return qhat, data[lower_idx], data[upper_idx]


def simulate(generator: SampleGenerator, batch: int, N: int, seed: int, alpha: float, delta: float) -> tuple[float, float, float]:
    numLargerThanPercentile = 0
    numMLM = 0
    numMLMLargerThanPercentile = 0
    for i in range(batch):
        samplesOne = generator.sample(N=N // 2, seed=seed + 2 * i)
        samplesTwo = generator.sample(N=N // 2, seed=seed + 2 * i + 1)
        samples = np.concatenate([samplesOne, samplesTwo], axis=0)

        sortedSamplesOne = np.sort(samplesOne)
        sortedSamplesTwo = np.sort(samplesTwo)
        sortedSamples = np.sort(samples)

        percentile = generator.getPercentile(alpha=alpha)

        bestSample = sortedSamples[-1]
        if bestSample > percentile:
            numLargerThanPercentile += 1

        bestSampleOne = sortedSamplesOne[-1]
        estimatedPercentile, lowInterval, highInterval = quantile_ci(
            samplesOne, p=0.9, alpha=0.05)
        print("estimated percentile = {}, low interval = {}, high interval = {}".format(
            estimatedPercentile, lowInterval, highInterval))
        if (highInterval - lowInterval) < 0.1:
            numMLM += 1
            if bestSampleOne > percentile:
                numMLMLargerThanPercentile += 1

    winRate = numLargerThanPercentile / batch
    winRateMLM = numMLMLargerThanPercentile / numMLM
    rateMLM = numMLM / batch
    return winRate, winRateMLM, rateMLM


def main():
    # generator = GaussianSampleGenerator(mu=0, sigma=1)
    generator = UniformSampleGenerator(a=0, b=1)

    """
    winRates = []
    winRatesMLM = []
    deltas = [2, 3, 4, 5, 6]
    for delta in deltas:
        winRate, winRateMLM, rateMLM = simulate(
            generator=generator, batch=1000, N=20, seed=42, alpha=0.9, delta=delta
        )
        winRates.append(winRate)
        winRatesMLM.append(winRateMLM)
    plt.plot(deltas, winRates, color="#EA8379", marker="s", label="best-of-N")
    plt.plot(deltas, winRatesMLM, color="#299D8F", marker="d", label="ours")
    plt.xlabel("deltas")
    plt.legend()
    plt.grid()
    plt.savefig("visualizations/win_rate_across_deltas_gaussian.png")
    plt.close()
    """

    winRates = []
    winRatesMLM = []
    sampleNums = [5, 10, 20, 30, 40, 50, 70, 100]
    for sampleNum in sampleNums:
        winRate, winRateMLM, rateMLM = simulate(
            generator=generator, batch=1000, N=sampleNum, seed=42, alpha=0.97, delta=5
        )
        winRates.append(winRate)
        winRatesMLM.append(winRateMLM)
    plt.plot(sampleNums, winRates, color="#EA8379",
             marker="s", label="best-of-N")
    plt.plot(sampleNums, winRatesMLM, color="#299D8F",
             marker="d", label="ours")
    plt.xlabel("num of samples")
    plt.legend()
    plt.grid()
    plt.savefig("visualizations/win_rate_across_sample_nums_gaussian.png")
    plt.close()

    winRate, winRateMLM, rateMLM = simulate(
        generator=generator, batch=1000, N=20, seed=42, alpha=0.9, delta=3.5)
    print("winRate = {}, winRateMLM = {}, rateMLM = {}".format(
        winRate, winRateMLM, rateMLM))


if __name__ == "__main__":
    main()
