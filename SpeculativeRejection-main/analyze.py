import matplotlib.pyplot as plt
import numpy as np
import re

pat = re.compile(
    r'REWARD OF BEST RESPONSE:\s*([-+]?\d*\.?\d+(?:[eE][+-]?\d+)?)')

nums_bon = []
with open("bon.log", "r", encoding="utf-8") as f:
    for line in f:
        for m in pat.finditer(line):
            nums_bon.append(float(m.group(1)))

nums_eon = []
with open("eon.log", "r", encoding="utf-8") as f:
    for line in f:
        for m in pat.finditer(line):
            nums_eon.append(float(m.group(1)))

stop_batches = []
with open("stopped_batches.txt", "r") as f:
    for line in f:
        stop_batches.append(int(line))

nums_bon = nums_bon[:len(nums_eon)]

print("nums_eon - nums_bon = {}".format(np.array(nums_eon) - np.array(nums_bon)))
print(np.mean(stop_batches))

plt.plot(range(len(nums_eon)), (np.array(nums_eon) - np.array(nums_bon)) /
         np.array(nums_bon), label="reward")
plt.plot(range(len(nums_eon)), (np.array(
    stop_batches) - 5) / 5, label="stopped batch")
plt.legend()
plt.savefig("analyze.png")
