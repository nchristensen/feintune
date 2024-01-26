import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hjson
import seaborn as sns

directories = ["autotuning_files_order_" + str(num) for num in range(1,6)]
filebases = ["ab50b874b437ba1dc496c99c8a74d6185341672e7404b3c9411882a2f8ada676",
             "a0ac1ae756d309fde2ce3736b9b0fd4d6d4f3e4e4276eae9730648a8e674d5e2",
             "a3e4e3ca2475516a1f2596cf83cb451c98a19cff2a5d7b980b009f75bd1e12ed",
             "aaeaae087dfdb71f0b8671ea6e027e91fb059ef933e5ed47a2e2bf4bf22e3a5d",
             "eb2e47a28ff6a217e95be310c8a90bdcee87b9acd91a1c57df2d0717485be50e"]

#directory = "autotuning_files_order_1"
#filebase = "ab50b874b437ba1dc496c99c8a74d6185341672e7404b3c9411882a2f8ada676"
#directory = "autotuning_files_order_5"
#filebase = "eb2e47a28ff6a217e95be310c8a90bdcee87b9acd91a1c57df2d0717485be50e"

f, axes = plt.subplots(nrows=3,ncols=2,sharex=False,sharey=False,figsize=[6.5,7.5])
for ax_ind, (directory, filebase) in enumerate(zip(directories, filebases)):
    csv_filepath = directory + "/" + filebase + ".csv"
    json_filepath = directory + "/" + filebase + "_default.hjson"

    df = pd.read_csv(csv_filepath)
    fp = open(json_filepath)
    default_time = (hjson.load(fp))["data"]["avg_time"]
    fp.close()
    print(default_time)
    df = df[df["RUNTIME"] < 999]

    means = []
    mins = []
    data = []
    #plt.subplots(
    #plt.figure()
    for bs in range(1,8+1):
        sub_df = df
        sub_df = sub_df[sub_df["batch_size"] == bs]
        sub_df.prefetch = sub_df.prefetch.astype(bool)
        #sub_df = sub_df[sub_df["RUNTIME"] <= default_time]
        #print(sub_df)
        min_time = sub_df["RUNTIME"].min()
        mean_time = sub_df["RUNTIME"].mean()
        median_time = sub_df["RUNTIME"].median()
        max_time = sub_df["RUNTIME"].max()
        std_time = sub_df["RUNTIME"].std()
        print(median_time, std_time)
        data.append((min_time, median_time, mean_time, max_time,std_time))
        mins.append(min_time)

    #sub_df = df[df["prefetch"] == 1]
    sns.violinplot(data=df, x="batch_size", y="RUNTIME", inner="points", cut=0, ax=axes.flat[ax_ind],log_scale=True)#, split=False)
    axes.flat[ax_ind].set(xlabel=None,ylabel=None)
    #if ax_ind < 5:
    #    axes.flat[ax_ind].get_legend().remove()

    #plt.violinplot(sub_df["RUNTIME"], positions=[bs])
        #std = sub_df["RUNTIME"].std()
        #print(mean, std)
    #plt.show()

    data = np.array(data)
    axes.flat[ax_ind].semilogy(range(0,8),data[:,0],label="Minimum")
    axes.flat[ax_ind].semilogy(range(0,8),data[:,1], label="Median")
    axes.flat[ax_ind].semilogy(range(0,8),data[:,2], label="Mean")
    pn = ax_ind + 1
    axes.flat[ax_ind].set_title(f"Polynomial order {pn}", fontsize=12)
    #plt.errorbar(range(1,9), data[:,2],yerr=data[:,4], capsize=10, color="k")
    #plt.plot(data[:,3])

axes.flat[-1].legend(handles=axes.flat[0].get_legend_handles_labels()[0], fancybox=False, shadow=False, bbox_transform=f.transFigure, loc="center", fontsize=11)
axes.flat[-1].axis('off')

f.supylabel("Kernel execution time (s)",fontsize=12)
f.supxlabel("Sub-batch size",fontsize=12)
f.suptitle("Effect of sub-batch size on OpenCL kernel execution time",fontsize=14)
f.tight_layout()
#plt.show()
f.savefig("subbatch_size.pdf", bbox_inches="tight")

## Look at prefetching

import os
f, axes = plt.subplots(nrows=3,ncols=2,sharex="all",sharey=False,figsize=[6.5,7.5])
for ax_ind, directory in enumerate(directories):

    df = None
    files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    for file in files:
        current_data = pd.read_csv(directory+"/"+file,header=0)
        if df is not None:
            #print(df["RUNTIME"].type())
            df = pd.concat([df,current_data],ignore_index=True)
        else:
            df = current_data

    df = df[df["RUNTIME"] != "RUNTIME"]
    #df = df.convert_dtypes()
    print(df["RUNTIME"])
    df.RUNTIME = df.RUNTIME.astype(float)
    df.prefetch = df.prefetch.astype(bool)
    #df.ji = df.ji.astype(bool)
    #for entry in df["RUNTIME"]:
    #    if not isinstance(entry, (int,float,np.float32)):
    #        print(type(entry),entry)
    #exit()
    df = df[df["RUNTIME"] < 999]
    #df["dummy"] = ""
    #print(df)

    #plt.figure()
    #ax = plt.axes()
    vals = [0,1]
    sns.violinplot(data=df, y="RUNTIME", hue="prefetch", inner="box", ax=axes.flat[ax_ind],log_scale=True,split=True)
    axes.flat[ax_ind].set(xlabel=None,ylabel=None)
    handles, previous_labels = axes.flat[ax_ind].get_legend_handles_labels()
    #axes.flat[ax_ind].legend(labels=["Prefetching disabled", "Prefetching enabled"], handles=handles)
    pn = ax_ind + 1
    axes.flat[ax_ind].set_title(f"Polynomial order {pn}", fontsize=12)

    if ax_ind < 5:
        axes.flat[ax_ind].get_legend().remove()

axes[-1][-1].legend(handles=handles, labels=["Prefetching disabled", "Prefetching enabled"], fancybox=False, shadow=False, bbox_transform=f.transFigure, loc="center", fontsize=11)
#handles, previous_labels = axes[-1][-1].get_legend_handles_labels()
#axes[-1][-1].legend(labels=["Prefetching disabled", "Prefetching enabled"], handles=handles)

axes[-1, -1].axis('off')
f.supylabel("Kernel execution time (s)",fontsize=12)
#f.supxlabel("Sub-batch size",fontsize=12)
f.suptitle("Effect of prefetching on OpenCL kernel execution time",fontsize=14)
f.tight_layout()
#plt.show()
f.savefig("prefetching_enabled.pdf", bbox_inches="tight")


## Look at unrhint

import os
f, axes = plt.subplots(nrows=3,ncols=2,sharex="all",sharey=False,figsize=[6.5,7.5])
for ax_ind, directory in enumerate(directories):

    df = None
    files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    for file in files:
        current_data = pd.read_csv(directory+"/"+file,header=0)
        if df is not None:
            #print(df["RUNTIME"].type())
            df = pd.concat([df,current_data],ignore_index=True)
        else:
            df = current_data

    df = df[df["RUNTIME"] != "RUNTIME"]
    #df = df.convert_dtypes()
    #print(df["RUNTIME"])
    df.RUNTIME = df.RUNTIME.astype(float)
    #df.prefetch = df.prefetch.astype(bool)
    df.ji = df.ji.astype(bool)
    #for entry in df["RUNTIME"]:
    #    if not isinstance(entry, (int,float,np.float32)):
    #        print(type(entry),entry)
    #exit()
    df = df[df["RUNTIME"] < 999]
    #df["dummy"] = ""
    #print(df)

    #plt.figure()
    #ax = plt.axes()
    vals = [0,1]
    sns.violinplot(data=df, y="RUNTIME", hue="ji", inner="box", ax=axes.flat[ax_ind],log_scale=True,split=True)
    axes.flat[ax_ind].set(xlabel=None,ylabel=None)
    handles, previous_labels = axes.flat[ax_ind].get_legend_handles_labels()
    #axes.flat[ax_ind].legend(labels=["Prefetching disabled", "Prefetching enabled"], handles=handles)
    pn = ax_ind + 1
    axes.flat[ax_ind].set_title(f"Polynomial order {pn}", fontsize=12)

    if ax_ind < 5:
        axes.flat[ax_ind].get_legend().remove()

axes[-1][-1].legend(handles=handles, labels=["unr_hint disabled", "unr_hint enabled"], fancybox=False, shadow=False, bbox_transform=f.transFigure, loc="center", fontsize=11)
#handles, previous_labels = axes[-1][-1].get_legend_handles_labels()
#axes[-1][-1].legend(labels=["Prefetching disabled", "Prefetching enabled"], handles=handles)

axes[-1, -1].axis('off')
f.supylabel("Kernel execution time (s)",fontsize=12)
#f.supxlabel("Sub-batch size",fontsize=12)
f.suptitle("Effect of unr_hint on OpenCL kernel execution time",fontsize=14)
f.tight_layout()
#plt.show()
f.savefig("unr_hint_enabled.pdf", bbox_inches="tight")


plt.show()

#print(df)

#print(df)

