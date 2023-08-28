# Write best result to hjson file
#import csv
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
kernel_id = "7344480911b9231871c60bf856e21f6a3d3a2a30f1d106646db2dab7e5deb2ef"

strategies = ["rf", "gbrt", "dummy"]#, "et", "gp"]

base_directory = "./pickled_programs_prediction_order_3"

from utils import load_hjson
data = load_hjson(base_directory + "/four_axis_hjson_dummy" + "/" + kernel_id + ".hjson")["data"]
roofline_flop_rate = data["roofline_flop_rate"]
flops = data["flops"]

plt.figure()

labels = {"rf": "Random Forest", "dummy": "Random sampling", "gbrt": "Gradient Boosted Regression Trees"}


for strategy in strategies:

    csv_filename = base_directory + "/four_axis_hjson_" + strategy + "/" + kernel_id + ".csv"
    dataframe = read_csv(csv_filename)["objective"].cummin()
    print(dataframe)
    plt.plot(np.arange(1, len(dataframe[:]) + 1), flops/dataframe[:]/roofline_flop_rate, label=labels[strategy])

plt.plot(np.arange(1, len(dataframe[:] + 1)), 0.5168756271127658*np.ones((100,)), "k--", label="Exhaustive search maximum")

plt.xlabel("Number of trials", fontsize=13)
plt.ylabel("Cumulative maximum fraction of roofline", fontsize=13)
plt.suptitle("Convergence of ytopt search strategies", fontsize=14)
plt.title("Order 3 divergence subkernel - ~300,000 elements - AMD MI100", fontsize=12)
#plt.rc('axes', fontsize=13)
#plt.rc('axes', titlesize=14)
#plt.rc('xtick', fontsize=12)
#plt.rc('ytick', labelsize=12)
#plt.rc('legend', fontsize=12)
plt.legend(fontsize=11)

plt.show()
#plt.pause()
