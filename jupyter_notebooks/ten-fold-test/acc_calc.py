import re
import os
import numpy as np

folders = [
    "/base/tool",
    "/base/spec",
    "/base/chip",
    "/base/multi_ts",
    "/base/multi_sc",
    "/base/multi_tc",
    "/base/multi_tsc",
]

cwd = os.getcwd()

for p in folders:
    p = cwd + p
    print("For path: " + p)
    accuracies = []
    try:
        with open(p + "/stdout.txt") as f:
            for line in f:
                if "Test accuracy T" in line:
                    s = re.findall(r"[+-]?[.]?[0-9]+", line)
                    tool = s[0]
                    acc = float(s[1] + s[2])
                    print("    Tool: ", tool, " accuracy: ", acc)
                    accuracies.append(acc)
        print("mean: ", np.mean(accuracies))
        print("std: ", np.std(accuracies))
    except:
        print("skip... file not found...")
    print("---------------------------------")
