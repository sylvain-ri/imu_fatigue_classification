# #######################################################################
# Imports
import logging
import os

import math
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from numpy.fft import fft
import numpy as np
from scipy import fftpack
from scipy.signal import blackman
from pprint import pprint



# #######################################################################
# #######################       Parameters        #######################
# #######################################################################
DEBUG = 2
SKIP_BEG_SEC = 1        # remove what happens when the recording on the smartphone is started and put into pocket
SKIP_END_SEC = 1        # same, for the end, when removed from pocket
SKIP_CHANGE_SEC = 0.8     # same, between transitions
THRESHOLD = 1         # threshold to cut the data
LABELS = {"none": -1, "standing": 0,
          "walk_rested": 1, "walk_tired": 2, "walk_very_tired": 3}

# Holders:
all_frames = []
frequencies = []
freq = 0

# Setup
col_to_plot = {1: ["acc_x", "acc_y", "acc_z"],
               2: ["acc_derivative_x", "acc_derivative_y", "acc_derivative_z"],
               3: ["acc_sum", "acc_sum_smoth"],
               4: ["acc_tot", "acc_deriv_tot"],
               5: ["gyro_x", "gyro_y", "gyro_z"]}


# #######################################################################
# #######################         load CSV        #######################
# #######################################################################
def load_csv(file_name):
    # Get read of first line (sep=;) when reading the csv file
    data = pd.read_csv(file_name, sep=";", skiprows=1, index_col=None)
    data["labels"] = LABELS["none"]
    return data


# #######################################################################
# #######################       main function     #######################
# #######################################################################
def clean_data(data, tiredness_state):
    # Set the datetime as index
    data.date_time = pd.to_datetime(data["date_time"], format='%Y-%m-%d %H:%M:%S:%f')
    # data["index"] = range(data.shape[0])
    # data.set_index("index")
    # time_ms converted into seconds
    data.time_ms = data.time_ms / 1000
    # find total time, sampling frequency
    logging.debug(f'starts at {data["date_time"][0]}, ends at {data["date_time"].iloc[-1]}')
    total_time = (data["date_time"].iloc[-1] - data["date_time"][0])
    frequency = int(round(len(data) / total_time.seconds, 0))
    logging.debug(len(data))
    logging.debug(frequency)


    # #######################################################################
    # #####################       PRINT AND PLOT       ######################
    # #######################################################################
    logging.info("Printing section")
    # Print some stuff
    if DEBUG > 6:
        print(data.head())
        print(data.columns)
        print(data.dtypes)


    # Plotting
    if DEBUG > 7:
        columns = ["acc_x", "acc_y", "acc_z", "grav_x", "grav_y", "grav_z",
                   "lin_acc_x", "lin_acc_y", "lin_acc_z",
                   "gyro_x", "gyro_y", "gyro_z",
                   "mag_x", "mag_y", "mag_z", "sound_lvl", ]
        col_to_plot = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z", ]
        plt.figure(f"{index}- ROW data")
        plt.plot(data.time_ms, data[col_to_plot])
        plt.legend(col_to_plot)
        plt.tight_layout()
        plt.show()



    # #######################################################################
    # #######################         FEATURES        #######################
    # #######################################################################
    logging.info("Feature engineering section")

    # window filter
    # add each difference on a 0.2 sec window
    # if that sum of noise is low, cut before, define it as standing state.
    # If DEBUG
    if DEBUG > 3:
        plt.figure(f"{index}- Smooth and acc variation")

    # Smoothing each acc axis
    axis = ('x', 'y', 'z')
    for xyz in axis:
        acc_smooth_xyz = f"acc_smooth_{xyz}"
        acc_derivative_xyz = f"acc_derivative_{xyz}"
        # Smooth acc on xyz axis
        data[acc_smooth_xyz] = data[f"acc_{xyz}"].rolling(round(0.2*frequency), win_type="hamming", center=True).sum()
        # compute derivative of each axis
        data[acc_derivative_xyz] = abs(data[acc_smooth_xyz].shift(1) - data[acc_smooth_xyz].shift(-1))
        if DEBUG > 5:
            plt.plot(data.time_ms, data[acc_smooth_xyz], label=acc_smooth_xyz)

    # sum of derivatives of each axis
    data["acc_tot"] = list(np.sqrt(data["acc_x"]**2 + data["acc_y"]**2 + data["acc_z"]**2))
    data["acc_deriv_tot"] = list(np.sqrt(data["acc_derivative_x"]**2 + data["acc_derivative_y"]**2 + data["acc_derivative_z"]**2))

    data["acc_sum"] = data["acc_derivative_x"] + data["acc_derivative_y"] + data["acc_derivative_z"]
    data["acc_sum_smoth"] = data["acc_sum"].rolling(round(0.8*frequency), win_type="hamming", center=True).sum() \
                            * 2 / round(0.8*frequency)


    # If DEBUG
    if DEBUG > 3:
        # plt.plot(data.time_ms, data["acc_sum"], label="acc_sum")
        plt.plot(data.time_ms, data["acc_sum_smoth"], label="acc_sum_smoth")
        plt.tight_layout()
        plt.legend()
        plt.show()

    #
    # ##################################################################
    # skip beginning and end of files
    # Labels
    # ##################################################################
    data.drop(range(SKIP_BEG_SEC*frequency), inplace=True)
    data.index = range(data.shape[0])

    # Ensure that we found the resting state
    i0_set_rest = data[data["acc_sum_smoth"] < THRESHOLD].index[0]
    data.drop(data.index[range(i0_set_rest)], inplace=True)      # remove beginning with noise
    data.index = range(data.shape[0])
    i0_set_rest = 0
    # Label until the end of first rest
    i1_rest_walk = data.loc[data.index > i0_set_rest + SKIP_CHANGE_SEC*frequency, "acc_sum_smoth"][data["acc_sum_smoth"] > THRESHOLD].index[0]
    data.loc[data.index <= i1_rest_walk, "labels"] = LABELS["standing"]
    # Label until the end of first walking
    i2_walk_rest = data.loc[data.index > i1_rest_walk + SKIP_CHANGE_SEC*frequency, "acc_sum_smoth"][data["acc_sum_smoth"] < THRESHOLD].index[0]
    data.loc[(i1_rest_walk <= data.index) & (data.index < i2_walk_rest), "labels"] = LABELS[tiredness_state]
    # Label until the turn
    i3_rest_turn = data.loc[data.index > i2_walk_rest + SKIP_CHANGE_SEC*frequency, "acc_sum_smoth"][data["acc_sum_smoth"] > THRESHOLD].index[0]
    data.loc[(i2_walk_rest <= data.index) & (data.index < i3_rest_turn), "labels"] = LABELS["standing"]
    # Label until rest
    i4_turn_rest = data.loc[data.index > i3_rest_turn + SKIP_CHANGE_SEC*frequency, "acc_sum_smoth"][data["acc_sum_smoth"] < THRESHOLD].index[0]
    data.loc[(i3_rest_turn <= data.index) & (data.index < i4_turn_rest), "labels"] = LABELS["none"]
    # Label until 2nd walk
    i5_rest_walk = data.loc[data.index > i4_turn_rest + SKIP_CHANGE_SEC*frequency, "acc_sum_smoth"][data["acc_sum_smoth"] > THRESHOLD].index[0]
    data.loc[(i4_turn_rest <= data.index) & (data.index < i5_rest_walk), "labels"] = LABELS["standing"]
    # Label until 2nd rest
    i6_walk_rest = data.loc[data.index > i5_rest_walk + SKIP_CHANGE_SEC*frequency, "acc_sum_smoth"][data["acc_sum_smoth"] < THRESHOLD].index[0]
    data.loc[(i5_rest_walk <= data.index) & (data.index < i6_walk_rest), "labels"] = LABELS[tiredness_state]
    # Label until noise
    i7_rest_noise = data.loc[data.index > i6_walk_rest + SKIP_CHANGE_SEC*frequency, "acc_sum_smoth"][data["acc_sum_smoth"] > THRESHOLD].index[0] - 10
    data.loc[(i6_walk_rest <= data.index) & (data.index < i7_rest_noise), "labels"] = LABELS["standing"]
    # Drop end part
    data.drop(data.loc[(i7_rest_noise <= data.index)].index, inplace=True)

    # Collect indexes
    split_indexes = [i0_set_rest, i1_rest_walk, i2_walk_rest, i3_rest_turn, i4_turn_rest,
                     i5_rest_walk, i6_walk_rest, i7_rest_noise]

    if DEBUG > 2:
        print(split_indexes)
    if DEBUG > 5:
        print(data["labels"].sample(10))

    # Plots
    if DEBUG > 2 and index in (3, 14):
        plt.figure(f"{index}- Gyro and labels")
        # plt.plot(data.time_ms, data["acc_sum"], label="acc_sum")
        plt.plot(data.time_ms, data["gyro_x"], label="acc_sum")
        plt.plot(data.time_ms, data["gyro_y"], label="acc_sum")
        plt.plot(data.time_ms, data["gyro_z"], label="acc_sum")
        # plt.plot(data.time_ms, data["acc_sum_smoth"], label="acc_sum_smoth")
        plt.plot(data.time_ms, data["labels"], label="labels")
        plt.tight_layout()
        plt.legend()
        # plt.show()

    # Split each part
    split_frame = []
    for ind_low, ind_high in zip(split_indexes[:-1], split_indexes[1:]):
        split_frame.append({"label": data["labels"].iloc[ind_low],
                            "frame": data[(ind_low <= data.index) & (data.index < ind_high)]})
        if DEBUG > 6:
            print("ind low and high : ", ind_low, ind_high, data.shape)
            print(data["labels"].iloc[ind_low])

    return split_frame, frequency


# #######################################################################
# #######################          KERAS          #######################
# #######################################################################
if False:
    import keras
    from keras.layers import GRU, Dense
    from keras.models import Sequential
    x = (nb_of_data, nb_of_samples, nb_of_features)
    y = nb_of_data * 1

    model = Sequential(input_shape={timesteps, n_features})
    model.add(GRU(256))
    model.add(Dense(1))

    model.compile(optimizer='adam')

    model.fit(x, y)


# #######################################################################
# #######################          SETUP          #######################
# #######################################################################
# todo add label for walk speed  standing/slow/medium/fast
def setup_files():
    # Logging Settings
    logging.basicConfig(level=logging.INFO)
    logging.info("Setup section")

    # Folder and Files
    folder = "D:/Drive/Singapore/Courses/CS6206-HCI Human Computer Interaction/Project/Data"
    os.chdir(folder)
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    logging.info(f"found {len(files)} files in folder {folder}")

    return files


# #######################################################################
# #######################          MAIN           #######################
# #######################################################################
# for index, file_name in enumerate([files[5], files[17]]):
def files_load_clean_to_feat(files):
    global all_frames
    global THRESHOLD

    for index, file_name in enumerate(files):

        if index not in (20,):
            # Fixing early experiments
            if index == 0:              THRESHOLD = 5.67
            elif index == 3:            THRESHOLD = 3.5
            elif index in range(5):     THRESHOLD = 4
            else:                       THRESHOLD = 2

            # if index in (0, 3): DEBUG = 4
            # else:  DEBUG = 0

            state = "none"
            if index in range(0, 6):      state = "walk_rested"
            elif index in range(6, 12):   state = "walk_tired"
            elif index in range(12, 18):  state = "walk_very_tired"

            print(f"\n *** File number {index} ***")
            frame = load_csv(file_name)
            print("data loaded")
            print(frame.shape)
            split_frame, frequency = clean_data(frame, state)
            print(frame.shape)
            print("data cleaned")

            all_frames.extend(split_frame)
            frequencies.append(frequency)

    global freq
    freq = sum(frequencies) / len(frequencies)


def plot_stuff():
    # whyyyyyy is plt. blocking...

    print("\n")
    print("Preprocessing done, continuing on single dataset extraction")
    print("len(all_frames)", len(all_frames))
    # pprint(all_frames)


    # Compute fft on all subsets
    col_to_plot = ["acc_derivative_z",]

    plt.figure(f"FFT comparison {col_to_plot}")
    plt.xlabel('Frequency in Hertz [Hz]')
    plt.ylabel('Frequency Domain Magnitude (Spectrum)')
    plt.xlim([-0.2, 10])
    red_patch = mpatches.Patch(color='red', label='walk_very_tired')
    green_patch = mpatches.Patch(color='green', label='walk_rested')
    blue_patch = mpatches.Patch(color='blue', label='standing')
    plt.legend(handles=[red_patch, green_patch, blue_patch])

    plt.show()
    plt.close("all")


def fft_frames():
    print("Compute the fft for some columns")

    col_to_fft = col_to_plot[1] + col_to_plot[3] + col_to_plot[4] + col_to_plot[5]

    for index, dat in enumerate(all_frames):

        if dat["label"] != LABELS["none"] and dat["label"] != LABELS["walk_tired"]:

            # Find the state
            state = list(LABELS.keys())[list(LABELS.values()).index(dat['label'])]

            # FFT will be hold under dat["fft"]
            dat["fft"] = {}

            for column in col_to_fft:

                signal = dat["frame"][column]

                f_s = freq
                X = fftpack.fft(signal)
                freqs = fftpack.fftfreq(len(signal)) * f_s

                dat["frame"][f"fft_{column}"] = freqs

                # X = np.round(abs(X), 2)
                # print("Printing x, then X reversed")
                # print(X[:10])
                #
                # X[0] = X[0] / 2
                # X_pos = X[math.floor(len(X)/2):]
                # ind_max = np.argpartition(abs(X_pos), -5)[-5:]
                # dat["fft"][column] = [(round(i/freq, 2), X[ind_max]) for i in ind_max]

                # print(X_pos[:10])
                # print(ind_max[:10])
                # print(dat["fft"][column])

                if DEBUG > 1:
                    # print("sorted X : ", sorted(X_pos)[:5])
                    # pprint(("dat['fft'][column]", dat["fft"][column]))
                    pass

                if DEBUG > 2:
                    if state == "standing":             colour = 'b'
                    elif state == "walk_rested":        colour = 'g'
                    elif state == "walk_very_tired":    colour = 'r'
                    plt.plot(freqs, abs(X), colour)

                # N = dat["frame"][column].shape[0]
                # T = float(N) / freq
                # x = np.linspace(0.0, N * T, N)
                # yf = scipy.fftpack.fft(dat["frame"][column])
                # xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
                #
                # dat["fft"][column] = xf, 2.0/N * np.abs(yf[:N//2])

            # pprint(dat["fft"][column])
            # for i in range(1, 4):
            #     plt.figure(f"index {index}: {state} - FFT data - {col_to_plot[i]}")
            #     plt.plot([abs(dat["fft"][x]) for x in col_to_plot[i]])
            #     plt.legend(col_to_plot[i])
            #     plt.tight_layout()

            # plt.show()
    print("Done")


# #######################################################################
# #######################         sklearn         #######################
# #######################################################################
def classification():
    logging.info("sklearn section")

    # todo feed into random forest
    if False:
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(x, y)
        rf.predict()






































