# buddy.ai-assignment-3
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

data = pd.read_excel("assignment 3 data excel.xlsx")
data


def grid_search(n,m):
    mae = []
    thetas = []
    for a in range(n,m+1,1):
        for b in range(n,m+1,1):
            for c in range(n,m+1,1):
                thetas.append([a,b,c])
                y_pred = a*(data['x']**2) + b*data['x'] + c
                error = data['y'] - y_pred
                mae.append(np.mean(error))
    return mae,thetas

def find_optimal(mae,thetas):
    best_mae = min(list(map(abs,mae)))
    index_mae = mae.index(best_mae)
    return thetas[index_mae],best_mae

def plot_qudratic(optimal_theta,data):
    pred = optimal_theta[0]*data['x']**2 + optimal_theta[1]*data['x'] + optimal_theta[2]
    plt.figure(figsize=(10,10),dpi=80)
    sns.set_style("whitegrid")
    sns.lineplot('x','y',data=data)
    sns.lineplot(data['x'],pred,color='g')
    plt.text(-15,1500, 'Parabola $Y = 3x^2+x+5$', fontsize = 20, 
            bbox = dict(facecolor = 'green', alpha = 1.0))
    plt.title('BUDDY Assignment3')
    plt.show()
    
    def grid_search(n,m):
    mae = []
    thetas = []
    for a in range(n,m+1,1):
        for b in range(n,m+1,1):
            for c in range(n,m+1,1):
                thetas.append([a,b,c])
                y_pred = a*(data['x']**2) + b*data['x'] + c
                error = data['y'] - y_pred
                mae.append(np.mean(error))
    return mae,thetas

def find_optimal(mae,thetas):
    best_mae = min(list(map(abs,mae)))
    index_mae = mae.index(best_mae)
    return thetas[index_mae],best_mae

def plot_qudratic(optimal_theta,data):
    pred = optimal_theta[0]*data['x']**2 + optimal_theta[1]*data['x'] + optimal_theta[2]
    plt.figure(figsize=(10,10),dpi=80)
    sns.set_style("whitegrid")
    sns.lineplot('x','y',data=data)
    sns.lineplot(data['x'],pred,color='g')
    plt.text(-15,1500, 'Parabola $Y = 3x^2+x+5$', fontsize = 20, 
            bbox = dict(facecolor = 'green', alpha = 1.0))
    plt.title('BUDDY Assignment3')
    plt.show()
