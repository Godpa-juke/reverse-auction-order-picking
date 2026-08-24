from pathlib import Path
from datetime import datetime
import time
from glob import glob
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
import seaborn as sns


simTime = [392,316,325,314,329,406,335,330,327,320,383,283,299]
pcs = [4036,4008,4007,4018,4001,4010,4014,4005,4003,4032,4018,4007,4021]
merge_information = [[pcs[idx],simTime[idx]] for idx in range(len(simTime))]

productivity = [pcs[idx]/(simTime[idx]/60) for idx in range(len(simTime))]
EnvStr = ['10_1H_ALL','10_1H_BIG','10_1H_BIGAD','10_1H_BIGALL','10_1H_SMALL','10_3H_ALL','10_3H_BIG','10_3H_BIGAD','10_3H_BIGALL','10_3H_SMALL','15_NO_ALL','15_NO_BIG','15_NO_SMALL']
df_result = pd.DataFrame(merge_information,index=EnvStr)


productivity = [productivity[5],productivity[6],productivity[9]]
EnvStr = [EnvStr[5],EnvStr[6],EnvStr[9]]


# time.sleep(100)

time_str = str(datetime.now().strftime('%Y%m%d%H%M%S'))
save_path = './'+time_str
os.mkdir(save_path)



width = 2*len(EnvStr)
height = 10
sns.set(rc={'figure.figsize': (width, height)})
sns.set(font_scale=1)
plt.xticks(rotation = +45)
bar_plot = sns.barplot(x=EnvStr, y=productivity)
for point in bar_plot.patches:
    bar_plot.text(point.get_x() + point.get_width() / 2,
                  point.get_y() + point.get_height(),
                  f"{point.get_height():.2f}",
                  ha='center')
plt.savefig(save_path + '/' + time_str + '.png')
plt.cla()


