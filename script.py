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
import os
os.chdir("D:/Drive/Singapore/Courses/CS6206-HCI Human Computer Interaction/Project/Python")
from plot_sensors import *

pd.set_option('display.expand_frame_repr', False)
files = setup_files()
files_load_clean_to_feat(files)
fft_frames()

df = features_to_pandas()
rf_classification(test_ratio=0.1, trees=100)


plot_each_feat(2)
plot_fft(8)



# plt.semilogy(xf[1:N//2], 2.0/N * np.abs(signal[1:N//2]), '-b')


