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

# either
df = stats_to_pandas()
# or
fft_frames()
df = fft_features_to_pandas()

# then classification
features_col = df.columns[1:-1]
train_x, test_x, train_y, test_y = train_test_split(df[features_col], df['label'], test_size=0.25)

# Trying GradientBoostingClassifier
rf_classification(trees=100)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2,
                                 max_depth=1).fit(train_x, train_y)
clf.score(test_x, test_y)


plot_each_feat(2)
plot_fft(8)



# plt.semilogy(xf[1:N//2], 2.0/N * np.abs(signal[1:N//2]), '-b')


