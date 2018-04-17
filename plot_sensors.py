import logging
import os
from pprint import pprint

from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import pandas as pd


# #######################################################################
# #######################          SETUP          #######################
# #######################################################################

# Logging Settings
logging.basicConfig(level=logging.INFO)

# Folder and Files
folder = "D:/Drive/Singapore/Courses/CS6206-HCI Human Computer Interaction/Project/Data"
os.chdir(folder)
files = [f for f in os.listdir(folder) if f.endswith(".csv")]
print(f"found {len(files)} files in folder {folder}")


# Get read of first line (sep=;) when reading the csv file
data = pd.read_csv(files[12], sep=";", skiprows=1, index_col=None)

# Set the datetime as index
data.date_time = pd.to_datetime(data["date_time"], format='%Y-%m-%d %H:%M:%S:%f')
data.set_index("date_time")


# #######################################################################
# #####################       PRINT AND PLOT       ######################
# #######################################################################

# Print some stuff
if True:
    print(data.head())
    print(data.columns)
    print(data.dtypes)


# Plotting
if True:
    plt.plot(data[["acc_x", "acc_y", "acc_z", "grav_x", "grav_y", "grav_z",
                   "lin_acc_x", "lin_acc_y", "lin_acc_z",
                   "gyro_x", "gyro_y", "gyro_z",
                   "mag_x", "mag_y", "mag_z", "sound_lvl", ]])
    plt.show()


# #######################################################################
# #######################         FEATURES        #######################
# #######################################################################

# todo skip beginning and end of files
# todo break into multiple "steps"
# todo fft(), max acc, ...





# #######################################################################
# #######################         sklearn         #######################
# #######################################################################

# todo feed into random forest
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
