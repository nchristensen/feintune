import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#directory = ("./autotuning_files/")
directory = ("./autotuning_files_order_2/")
files = os.listdir(directory)
files = [file for file in files if str(file).endswith(".hjson")]
pids = [file[:-6] for file in files if not (str(file).endswith("full.hjson") or str(file).endswith("default.hjson"))]

def get_batch_size(batch_dict):
  transformations = batch_dict["transformations"]
  for t in transformations:
      if t[0] == "batch_einsums":
          return t[1][0]
  return None

from tagtune.utils import load_hjson

batch_dict_list = []
full_dict_list = []
default_dict_list = []
faster_dict_list = []
faster_dict_list_only_tuned = []
faster_dict_list_only_tuned_better = []
default_dict_list_only_worse = []
default_if_not_full = []
faster_dict_only_default_exists = []

for pid in pids:
  batch_dict_list.append(load_hjson(directory + pid + ".hjson")["data"])
  full_dict = None
  default_dict = None
  if os.path.exists(directory + pid + "_full.hjson"):
      full_dict = load_hjson(directory + pid + "_full.hjson")["data"]
      full_dict_list.append(full_dict)
  if os.path.exists(directory + pid + "_default.hjson"):
      default_dict = load_hjson(directory + pid + "_default.hjson")["data"]
      default_dict_list.append(default_dict)

    

  if default_dict is not None and full_dict is not None:
      if full_dict["avg_time"] < default_dict["avg_time"]:
          faster_dict_list.append(full_dict)
          faster_dict_list_only_tuned.append(full_dict)
          faster_dict_list_only_tuned_better.append(full_dict)
          default_dict_list_only_worse.append(default_dict)
          faster_dict_only_default_exists.append(full_dict)
      else:
          faster_dict_list.append(default_dict)
          faster_dict_list_only_tuned.append(default_dict)
          faster_dict_only_default_exists.append(default_dict)
  elif default_dict is not None:
      faster_dict_list.append(default_dict)
      faster_dict_only_default_exists.append(default_dict)
  elif full_dict is not None:
      faster_dict_list.append(full_dict)

batch_df = pd.DataFrame(data=batch_dict_list).fillna(0)
full_df = pd.DataFrame(data=full_dict_list).fillna(0)
default_df = pd.DataFrame(data=default_dict_list).fillna(0)
faster_df = pd.DataFrame(data=faster_dict_list).fillna(0)
faster_df_only_tuned = pd.DataFrame(data=faster_dict_list_only_tuned).fillna(0)
faster_df_only_tuned_better = pd.DataFrame(data=faster_dict_list_only_tuned_better).fillna(0)
default_df_only_worse = pd.DataFrame(data=default_dict_list_only_worse).fillna(0)
faster_df_only_default_exists = pd.DataFrame(data=faster_dict_only_default_exists).fillna(0)
#full_execution_times = (full_df["avg_time"].to_numpy() - full_df["device_latency"].to_numpy())
#full_execution_times = full_execution_times * (full_execution_times > 0).astype(np.float64)
full_execution_times = full_df["avg_time"]

for entry in full_df["frac_roofline_flop_rate"]:
    print(entry)
for entry in full_df["avg_time"]:
    print(entry)

"""
# Weight each kernel by own execution time
avg_roofline_full = np.average(full_df["frac_roofline_flop_rate"], weights=full_execution_times)
avg_roofline_default = np.average(default_df["frac_roofline_flop_rate"], weights=default_df["avg_time"])
avg_roofline_predicted = np.average(batch_df["frac_roofline_flop_rate"], weights=batch_df["avg_time"])
avg_roofline_faster = np.average(faster_df["frac_roofline_flop_rate"], weights=faster_df["avg_time"])
avg_roofline_faster_only_tuned = np.average(faster_df_only_tuned["frac_roofline_flop_rate"], weights=faster_df_only_tuned["avg_time"])
avg_roofline_faster_only_tuned_better = np.average(faster_df_only_tuned_better["frac_roofline_flop_rate"])#, weights=faster_df_only_tuned_better["avg_time"])
avg_roofline_default_only_worse = np.average(default_df_only_worse["frac_roofline_flop_rate"])#, weights=default_df_only_worse["avg_time"])
"""

# Weight each kernel by default transformed kernel execution time
from scipy.stats import gmean as average
avg_roofline_full = average(full_df["frac_roofline_flop_rate"], weights=full_execution_times)
avg_roofline_default = average(default_df["frac_roofline_flop_rate"], weights=default_df["avg_time"])
avg_roofline_predicted = average(batch_df["frac_roofline_flop_rate"], weights=batch_df["avg_time"])
avg_roofline_faster = average(faster_df["frac_roofline_flop_rate"], weights=faster_df["avg_time"])
avg_roofline_faster_only_tuned = average(faster_df_only_tuned["frac_roofline_flop_rate"], weights=faster_df_only_tuned["avg_time"])
avg_roofline_faster_only_tuned_better = average(faster_df_only_tuned_better["frac_roofline_flop_rate"])#, weights=faster_df_only_tuned_better["avg_time"])
avg_roofline_default_only_worse = average(default_df_only_worse["frac_roofline_flop_rate"])#, weights=default_df_only_worse["avg_time"])

#avg_max_roofline = np.average(np.maximum(default_df["frac_roofline_flop_rate"].to_numpy(), full_df["frac_roofline_flop_rate"]), weights=np.minimum(default_df["avg_time"], full_df["avg_time"]))

#print(avg_roofline_predicted, avg_roofline_full, avg_roofline_default, avg_roofline_faster, avg_roofline_faster_only_tuned, avg_roofline_faster_only_tuned_better, avg_roofline_default_only_worse)

print("Total execution time")

def get_weighted_avg_time(df):
    import json
    import os
    this_dir, this_filename = os.path.split(__file__)
    JSON_PATH = os.path.join(this_dir, "saved_call_count_1.json")
    d = load_hjson(JSON_PATH)

    weighted_times = []
    for name, time in zip(df["name"], df["avg_time"]):
        print(name)
        basename = name[:name.rfind("_")]
        if basename in d:
            weighted_times.append(d[basename]*time)
        elif name in d:
            weighted_times.append(d[name]*time)
        else:
            raise KeyError

    return weighted_times
        

default_df = default_df.assign(weighted_avg_time=pd.Series(get_weighted_avg_time(default_df)).values)
faster_df = faster_df.assign(weighted_avg_time=pd.Series(get_weighted_avg_time(faster_df)).values)

default_avg_time = np.sum(default_df["weighted_avg_time"])
faster_avg_time_total = np.sum(faster_df["weighted_avg_time"])
speedup = default_avg_time / faster_avg_time_total

tuning_potential = faster_df["weighted_avg_time"].to_numpy()*(1 - faster_df["frac_roofline_flop_rate"])
remaining_speedup = faster_avg_time_total/(faster_avg_time_total - np.sum(tuning_potential))



faster_df_only_tuned_better = faster_df_only_tuned_better.assign(weighted_avg_time=pd.Series(get_weighted_avg_time(faster_df_only_tuned_better)).values)
default_df_only_worse = default_df_only_worse.assign(weighted_avg_time=pd.Series(get_weighted_avg_time(default_df_only_worse)).values)


default_only_worse_avg_time = np.sum(default_df_only_worse["weighted_avg_time"])
faster_only_better_avg_time_total = np.sum(faster_df_only_tuned_better["weighted_avg_time"])

speedup2 = default_only_worse_avg_time / faster_only_better_avg_time_total

tuning_potential2 = faster_df_only_tuned_better["weighted_avg_time"].to_numpy()*(1 - faster_df_only_tuned_better["frac_roofline_flop_rate"])
remaining_speedup2 = faster_only_better_avg_time_total/(faster_only_better_avg_time_total - np.sum(tuning_potential2))


print("All kernels", np.sum(default_df["weighted_avg_time"]), faster_avg_time_total, speedup, remaining_speedup)
print("Sped up kernels", np.sum(default_df_only_worse["weighted_avg_time"]), faster_only_better_avg_time_total, speedup2, remaining_speedup2)

data_dict = {}
for entry in zip(default_df["name"],default_df["flops"], faster_df["avg_time"], faster_df["frac_roofline_flop_rate"], default_df["frac_roofline_flop_rate"]):
    data_dict[entry[0]] = entry

print(data_dict)
for entry in data_dict.values():
    print(entry)

#plt.plot(sorted(tuning_potential, reverse=True)/faster_avg_time_total)
#plt.show()

#ngroups = 2
#pos1 = np.arange(0, ngroups*len(default_df["avg_time"]), ngroups)
#pos2 = np.arange(1, ngroups*len(default_df["avg_time"]) + 1, ngroups)


#fix, ax = plt.subplots(nrows=2, ncols=1, layout="constrained")
#ax[0].bar(pos1, default_df.sort_values("avg_time", ascending=False)["frac_roofline_flop_rate"])
#ax[1].bar(np.arange(len(default_df["avg_time"])), sorted(default_df["avg_time"], reverse=True))
#ax[1].bar(np.arange(len(full_df["avg_time"])), sorted(default_df["avg_time"], reverse=True))

#"""
plt.semilogy(sorted(faster_df["avg_time"], reverse=True), label="Autotuned transformations")
plt.semilogy(sorted(default_df["avg_time"], reverse=True), label="Handtuned transformations")
#plt.semilogy(sorted(batch_df["avg_time_predicted"], reverse=True), label="Single batch predicted")
plt.legend(fontsize=10)

plt.suptitle("Ranked Y3 Prediction Smoke-test KS 3D Kernels", fontsize=13) 
plt.title("Order 2, Nvidia Titan V", fontsize=12)
plt.xlabel("Execution time ranking", fontsize=12)
plt.ylabel("Kernel execution time", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
#"""

plt.show()
