if False:
    import os
    folder = "D:/Drive/Singapore/Courses/CS6206-HCI Human Computer Interaction/Project/Data"
    for file in os.listdir(folder):
        pass

if False:
    s = input("input the string to parse")
    s = s.replace(";", '", "')
    print(s)

# How to call the main script from the console
from plot_sensors import *

files = setup_files()
files_load_clean_to_feat(files)
fft_frames()

one_frame = all_frames[5]['frame']


