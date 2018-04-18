# #######################################################################
# Imports
import logging
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label
from matplotlib import pyplot as plt
import pandas as pd
from pprint import pprint



# #######################################################################
# #######################       Parameters        #######################
# #######################################################################
DEBUG = 4
SKIP_BEG_SEC = 2    # remove what happens when the recording on the smartphone is started and put into pocket
SKIP_END_SEC = 3    # same, for the end, when removed from pocket



# #######################################################################
# #######################         load CSV        #######################
# #######################################################################
def load_csv(file_name):
    # Get read of first line (sep=;) when reading the csv file
    return pd.read_csv(file_name, sep=";", skiprows=1, index_col=None)


# #######################################################################
# #######################       main function     #######################
# #######################################################################
def clean_data(data):
    # Set the datetime as index
    data.date_time = pd.to_datetime(data["date_time"], format='%Y-%m-%d %H:%M:%S:%f')
    data.set_index("date_time")
    # time_ms converted into seconds
    data.time_ms = data.time_ms / 1000
    # find total time, sampling frequency
    logging.DEBUG(f'starts at {data["date_time"][0]}, ends at {data["date_time"].iloc[-1]}')
    total_time = (data["date_time"].iloc[-1] - data["date_time"][0])
    frequency = int(round(len(data) / total_time.seconds, 0))
    logging.DEBUG(len(data))
    logging.DEBUG(frequency)


    # #######################################################################
    # #####################       PRINT AND PLOT       ######################
    # #######################################################################
    logging.info("Printing section")
    # Print some stuff
    if DEBUG > 1:
        print(data.head())
        print(data.columns)
        print(data.dtypes)


    # Plotting
    if DEBUG > 2:
        columns = ["acc_x", "acc_y", "acc_z", "grav_x", "grav_y", "grav_z",
                   "lin_acc_x", "lin_acc_y", "lin_acc_z",
                   "gyro_x", "gyro_y", "gyro_z",
                   "mag_x", "mag_y", "mag_z", "sound_lvl", ]
        col_to_plot = ["acc_x", "acc_y", "acc_z"]
        plt.figure(f"{index}- All columns")
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
        data[acc_smooth_xyz] = data[f"acc_{xyz}"].rolling(round(0.1*frequency)).sum()
        # compute derivative of each axis
        data[acc_derivative_xyz] = abs(data[acc_smooth_xyz].shift(1) - data[acc_smooth_xyz].shift(-1))
        if DEBUG > 3:
            plt.plot(data.time_ms, data[acc_smooth_xyz], label=acc_smooth_xyz)

    # sum of derivatives of each axis
    data["acc_sum"] = data["acc_derivative_x"] + data["acc_derivative_y"] + data["acc_derivative_z"]

    # If DEBUG
    if DEBUG > 3:
        plt.plot(data.time_ms, data["acc_sum"], label="acc_sum")
        plt.tight_layout()
        plt.legend()
        plt.show()


    # todo skip beginning and end of files
    min_variation = data["acc_sum"].min





    #
    # todo break into multiple "steps"
    # todo fft(), max acc, ...
    # todo gravity direction compared to stationary state !
    # todo break into tired/not and also into standing/slow/medium/fast

    return



# #######################################################################
# #######################         sklearn         #######################
# #######################################################################
logging.info("sklearn section")

# todo feed into random forest
if False:
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(x, y)
    rf.predict()





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
# todo add label for walk speed
# todo add label for tired/not

# Logging Settings
logging.basicConfig(level=logging.INFO)
logging.info("Setup section")

# Folder and Files
folder = "D:/Drive/Singapore/Courses/CS6206-HCI Human Computer Interaction/Project/Data"
os.chdir(folder)
files = [f for f in os.listdir(folder) if f.endswith(".csv")]
logging.info(f"found {len(files)} files in folder {folder}")


# #######################################################################
# #######################          MAIN           #######################
# #######################################################################
for index, file_name in enumerate([files[5], files[17]]):
    print(f"*** File number {index} ***")
    data = load_csv(file_name)
    print("data loaded")
    clean_data(data)
    print("data cleaned \n")


















