import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import tfr_morlet
import multiprocessing as mp
import time
from mpl_toolkits.mplot3d import Axes3D
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

evokedFif = '/home/timb/camcan/proc_data/TaskSensorAnalysis_transdef/CC110033/transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo-ave.fif'
exampleEv = mne.read_evokeds( evokedFif ) [0]
exampleEv.pick_types(meg='mag')
info = exampleEv.info


layout = mne.channels.find_layout(info, ch_type='mag')  
positions = layout.pos   
channels = layout.names
chn_x = positions[:,0]
chn_y = positions[:,1]
dictx = {a:b for a,b in zip(channels,chn_x)}
dicty = {a:b for a,b in zip(channels,chn_y)}



#Create the data
sourceDir = '/media/NAS/bbrady/spectralEvents/'

csvFile = 'CC110033_mag_spectral_events.csv'
#csvFile = '_all_mag_channels_all_subjects_spectral_events.csv'

csvData = os.path.join(sourceDir, csvFile)
data = pd.read_csv(csvData) 

df1 = data.drop(data[data['Peak Frequency'] < 10].index)
df = df1.drop(df1[df1['Peak Frequency'] > 12].index)
#df = df[df['Channel'] == 'MEG0711']


df['x_pos'] = df['Channel'].map(dictx)
df['y_pos'] = df['Channel'].map(dicty)



df = df[['x_pos', 'Peak Time', 'y_pos']]
X = df.to_numpy()





def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(1, 1, 1 )
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        print(i)
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    #plt.xlim(-9., 5.)
    #plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')

plt.show()
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['x_pos'], df['y_pos'], df['Peak Time'], s=0.05)
#ax = fig.add_subplot(111)
#ax.scatter(df['Event Duration'], df['Peak Frequency'], s=0.0000005)
plt.xlim(0,1)
ax.set_xlabel("x pos")
ax.set_ylabel("y pos")
ax.set_title("All channels all participants")
plt.show()
'''

