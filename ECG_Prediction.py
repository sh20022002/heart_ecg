import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def getData(name):
    # paths to the csv and anotation files
    path = r'C:\Users\shmue\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Visual Studio Code\python\open_pojects\test_3.10\heart_ECG'
    path_csv = path + r'\mitbih_database\{}.csv'.format(name)
    path_anotation = path + r'\mitbih_database\{}annotations.txt'.format(name)
    # colums of the csv and anotation files
    cols = ['Time', 'Sample', 'Type', 'Sub', 'Chan', 'Num']
    # anomleis and their numaric value
    # anomlie = {'N': 0, 'L': -0.5, 'R': 0.5, 'A': -1, 'V':-1}
    # decarator function to run the functions that reads the data and louds it one epoch at a time 
    csv_data = pd.read_csv(path_csv)
    ann_data = pd.read_csv(path_anotation)
    return ann_data, csv_data



def create_image_ecg(data, data_a, name):
    # shows a section of the ecg with the anomalies on the matplotlib plot
    fs = 360
    Ts = 1/fs

    labels = data_a['Type']
    data['time'] = data['sample']*Ts
    plt.plot(data['time'], data['V5'])

    for i, label in enumerate(labels):
        plt.annotate(label, (data_a['Sample'][i]*Ts, data['V5'][data_a['Sample'][i]]))
                 
    plt.plot(data_a['Sample']*Ts, data['V5'].iloc[data_a['Sample']], label=data_a['Type'], marker='*', linestyle='None')
    plt.xlim((0, 20))
    plt.grid()
    plt.xlabel('time (ms)')
    plt.ylabel('ecg')
    plt.title(f'ECG_{name}')
    plt.show()
     
def image_plotly(data, data_a, name):
    # shows the ecg with the anomalies on the plot
    fs = 360
    Ts = 1/fs
    data['time'] = data['sample']*Ts
    data['V5'] = data['V5'] - 980
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['time'], y=data['V5'], mode='lines', name='V5', line=dict(color='blue')))
    fig.update_layout(xaxis_title='time (ms)', yaxis_title='ecg')
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()
    

def main():
    ann, csv = getData("100")
    # column_names = ann.columns
    # print(column_names)
    image_plotly(csv, ann, "100")
    


if __name__ == '__main__':
    main()