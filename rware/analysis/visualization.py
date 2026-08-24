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

warnings.filterwarnings(action='ignore')


def time_convert_minute(date, time_format):
    time_result = datetime.strptime(date, time_format)
    total_minute = 60 * time_result.hour + time_result.minute
    return total_minute


data_path = './result'
columns = list()
result_data = list()
for root, directories, files in os.walk(data_path):
    columns.append(root[9:])
    for file in files:
        file_path = root + '/' + file
        if '_agent.csv' in file:
            #             df_agent = pd.read_csv(file_path)
            #             df_agent.drop(['Unnamed: 0'], axis = 1, inplace = True)
            pass

        elif '_information.csv' in file:
            #             df_batch = pd.read_csv(file_path)
            #             df_batch.drop(['Unnamed: 0'], axis = 1, inplace = True)
            pass
        elif '_summary.csv' in file:
            df_summary = pd.read_csv(file_path)
            data_list = list(df_summary.iloc[:, 0])
            time_format = '%H : %M : %S'

            actual_time_result = time_convert_minute(data_list[1][22:], time_format)
            running_time_result = time_convert_minute(data_list[2][7:], time_format)
            box_hour_human = round(float(data_list[6][16:]), 2)
            human_average_moving = round(float(data_list[8][31:]))
            time_out = round(float(data_list[14][17:]))
            robot_average_moving = round(float(data_list[15][31:]))
            save_list = [actual_time_result, running_time_result, box_hour_human, human_average_moving,
                         robot_average_moving, time_out]
            result_data.append(save_list)


        elif '_zone.csv' in file:
            pass

# columns = columns[:]
df_result = pd.DataFrame(columns=columns)
df_result[''] = ['Actual_running_time(min)', 'Sim_running_time(min)', 'Box_hour_Human', 'Human_moving_average',
                 'Robot_moving_average', 'Time_out_cnt']


for idx, value in enumerate(columns):
    if value == '':
        pass
    else:
        df_result[value] = result_data[idx - 1]

time_str = str(datetime.now().strftime('%Y%m%d%H%M%S'))
save_path = './'+time_str
os.mkdir(save_path)


for i in range(len(df_result)):
    width = 2*len(df_result)
    height = 10
    sns.set(rc={'figure.figsize': (width, height)})
    plt.title(df_result.iloc[i][0])
    sns.set(font_scale=1)
    plt.xticks(rotation = +45)
    bar_plot = sns.barplot(x=columns[1:], y=list(df_result.iloc[i, 1:]))
    for point in bar_plot.patches:
        bar_plot.text(point.get_x() + point.get_width() / 2,
                      point.get_y() + point.get_height(),
                      f"{point.get_height():.2f}",
                      ha='center')
    plt.savefig(save_path + '/' + time_str + '_{}.png'.format(df_result.iloc[i, 0]))
    plt.cla()
