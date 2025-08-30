import matplotlib.pyplot as plt
import numpy as np

from sample_generator import SampleGenerator, UniformSampleGenerator, GaussianSampleGenerator
from typing import Tuple


def simulate(generator: SampleGenerator, batch: int, N: int, seed: int, alpha: float, delta: float) -> tuple[float, float, float]:
    numLargerThanPercentile = 0
    numMLM = 0
    numMLMLargerThanPercentile = 0
    for i in range(batch):
        samples = np.sort(generator.sample(N=N, seed=seed + i))
        mean = np.mean(samples)
        deviation = np.std(samples, ddof=1)
        bestSamples = samples[-1]
        worstSamples = samples[0]

        percentile = generator.getPercentile(alpha=alpha)
        if bestSamples >= percentile:
            numLargerThanPercentile += 1

        if (bestSamples - worstSamples) >= delta:
            numMLM += 1
            if bestSamples >= percentile:
                numMLMLargerThanPercentile += 1

    winRate = numLargerThanPercentile / batch
    winRateMLM = numMLMLargerThanPercentile / numMLM
    rateMLM = numMLM / batch
    return winRate, winRateMLM, rateMLM


def main():
    generator = GaussianSampleGenerator(mu=0, sigma=1)

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

    winRates = []
    winRatesMLM = []
    sampleNums = [5, 10, 20, 30, 40, 50, 70, 100]
    for sampleNum in sampleNums:
        winRate, winRateMLM, rateMLM = simulate(
            generator=generator, batch=1000, N=sampleNum, seed=42, alpha=0.9, delta=5
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
