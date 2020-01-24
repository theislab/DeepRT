import os
import numpy as np
import pandas as pd
import sys

path = os.path.join(sys.path[2], "thickness_map_calculation", "thickness_maps")
thickness_maps = [os.path.join(os.path.join(path, i)) for i in os.listdir(path)]

def total_variation(lbl):

    # removing empty margin around each thickness map
    start = 30
    end = 128 - 30

    # calculate total variation across first axis
    total_variation = np.nanmean(np.abs(np.ediff1d(lbl[start:end,start:end].flatten())))
    total_variation = total_variation / (np.nanmax(lbl) - np.nanmin(lbl))

    # calculate total variation across second axis
    total_variation_t = np.nanmean(np.abs(np.ediff1d(np.transpose(lbl[start:end,start:end]).flatten())))
    total_variation_t = total_variation_t / (np.nanmax(lbl) - np.nanmin(lbl))

    tv = (total_variation + total_variation_t) / 2.
    return tv

tv_log = [[],[]]
for map_path in thickness_maps:
    record = map_path.split("/")[-1]
    im_record = record.replace(".npy",".png")

    # filter criteria is 0.014
    map = np.load(map_path)
    tv = total_variation(map)
    tv_log[0].append(record)
    tv_log[1].append(tv)

# save total variation log to data frame
tv_pd = pd.DataFrame(tv_log).T.rename(columns={0:"id",1:"tv"})
tv_pd.to_csv("./tv_pd.csv")

