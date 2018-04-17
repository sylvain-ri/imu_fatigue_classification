# #######################################################################
# Imports
import logging
import os
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import pandas as pd
from pprint import pprint


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


# Get read of first line (sep=;) when reading the csv file
data = pd.read_csv(files[12], sep=";", skiprows=1, index_col=None)

# Set the datetime as index
data.date_time = pd.to_datetime(data["date_time"], format='%Y-%m-%d %H:%M:%S:%f')
data.set_index("date_time")
data.time_ms = data.time_ms / 1000
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
if False:
    print(data.head())
    print(data.columns)
    print(data.dtypes)


# Plotting
if False:
    columns = ["acc_x", "acc_y", "acc_z", "grav_x", "grav_y", "grav_z",
               "lin_acc_x", "lin_acc_y", "lin_acc_z",
               "gyro_x", "gyro_y", "gyro_z",
               "mag_x", "mag_y", "mag_z", "sound_lvl", ]
    col_to_plot = ["acc_x", "acc_y", "acc_z"]
    plt.plot(data.time_ms, data[col_to_plot])
    # todo remove borders of matplot
    plt.show()


# #######################################################################
# #######################         FEATURES        #######################
# #######################################################################
logging.info("Feature engineering section")

# todo skip beginning and end of files
# window filter
# add each difference on a 0.5 sec window
# if that sum of noise is low, cut before, define it as standing state.
axis = ('x', 'y', 'z')
for xyz in axis:
    acc_smooth_xyz = f"acc_smooth_{xyz}"
    acc_change_xyz = f"acc_change_{xyz}"
    # Smooth acc on xyz axis
    data[acc_smooth_xyz] = data[f"acc_{xyz}"].rolling(round(0.1*frequency)).sum()
    # compute how much acceleration is changing
    data[acc_change_xyz] = abs(data[acc_smooth_xyz].shift(1) - data[acc_smooth_xyz].shift(-1))
    plt.plot(data.time_ms, data[acc_smooth_xyz])

data["acc_sum"] = data["acc_change_x"] + data["acc_change_y"] + data["acc_change_z"]
plt.plot(data.time_ms, data["acc_sum"])
plt.tight_layout()
plt.show()




#
# todo break into multiple "steps"
# todo fft(), max acc, ...
# todo gravity direction compared to stationary state !
# todo break into tired/not and also into standing/slow/medium/fast



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
