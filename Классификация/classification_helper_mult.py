import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ipywidgets import interact, IntSlider,  FloatSlider
import numpy as np
import pandas as pd
from sklearn import datasets


def get_data(irises = ["iris setosa","iris virginica"]):
    
    iris_dict = {0:"iris setosa",1:"iris versicolor",2:"iris virginica"}
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = np.array([iris_dict[iris] for iris in iris.target])
    return X[np.isin(y,irises)], y[np.isin(y,irises)]

def print_data(X,Y,columns):
    dict_param = {"Длина чашелистника":0, "Ширина чашелистника":1,"Длина лепестка":2,"Ширина лепестка":3}
    N = np.array([dict_param[par] for par in columns])    
    df = pd.DataFrame(np.hstack((X[:,N], Y.reshape(-1,1))), columns=columns+['Вид Ириса'])
    display(df)


def create_base_plot():
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(8, 6), dpi=100)
    plt.grid()
    

def plot_sign():
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot([-22, 0, 0, 22], [-1, -1, 1, 1])
    plt.xlabel("t", fontsize=17)
    plt.ylabel("sign(t)", fontsize=17)
    plt.grid()
    plt.ylim([-2, 2])
    plt.xlim([-21, 21])
    plt.show()

def plot_data_with_gip(X,Y):

    teta_0 = FloatSlider(min=-1.8, max=0, step=0.3, value=-0.92533091, description='theta_0: ')
    teta_1 = FloatSlider(min=1, max=3, step=0.25, value=1.25, description='theta_1: ')
    teta_2 = FloatSlider(min=-5, max=-1, step=0.5, value=-2.5, description='theta_2: ')

    @interact(teta_0=teta_0, teta_1=teta_1, teta_2=teta_2)
    def _plot_data_with_gip(teta_0,teta_1,teta_2):
        create_base_plot()
        plt.scatter(X[:,0][Y=='iris setosa'], X[:,1][Y=='iris setosa'], label='iris setosa', c='g')
        plt.scatter(X[:,0][Y=='iris virginica'], X[:,1][Y=='iris virginica'], label='iris virginica', c='b')
        plt.plot(X[:,0], (teta_0 +teta_1*X[:,0])/-teta_2)
        plt.xlabel("Длина чашелистника, см", fontsize=17)
        plt.ylabel("Ширина чашелистника, см", fontsize=17)
        plt.legend(prop={'size': 12})
        plt.xlim([4, 8])
        plt.ylim([2, 4.5])
        plt.show()


def plot_data(X,Y,columns):
    dict_param = {"Длина чашелистника":0, "Ширина чашелистника":1,"Длина лепестка":2,"Ширина лепестка":3}
    N = [dict_param[par] for par in columns]
    create_base_plot()
    irises = set(Y)
    for iris in irises:
        plt.scatter(X[:,N[0]][Y==iris], X[:,N[1]][Y==iris], label=iris)
    plt.xlabel(f"{columns[0]}, см", fontsize=17)
    plt.ylabel(f"{columns[1]}, см", fontsize=17)
    plt.legend(prop={'size': 12})
    plt.show()
    
    
def plot_data_moon(X,Y):
    create_base_plot()
    irises = set(Y)
    for iris in irises:
        plt.scatter(X[:,0][Y==iris], X[:,1][Y==iris], label=iris)
    plt.xlabel(f"Признак 1", fontsize=17)
    plt.ylabel(f"Признак 2", fontsize=17)
    plt.legend(prop={'size': 12})
    plt.show() 

def plot_decision_regions(X, y,columns, classifier):
    # настроить генератор маркеров и палитру
    dict_param = {"Длина чашелистника":0, "Ширина чашелистника":1,"Длина лепестка":2,"Ширина лепестка":3}
    dict_ = {0:"iris setosa",1:"iris versicolor",2:"iris virginica"}
    N = [dict_param[par] for par in columns]
    create_base_plot()
    markers = ('s','x','o', '^','v')
    colors = ('red', 'blue', 'lightgreen','gray','суаn')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # вывести поверхность решения
    x1_min, x1_max = X[:,N[0]].min() - 1, X[:,N[0]].max() + 1
    x2_min, x2_max = X[:,N[1]].min() - 1, X[:,N[1]].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,0.02),
                           np.arange(x2_min,x2_max,0.02))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(0, xx1.max())
    plt.ylim(0, xx2.max())
    # показать все образцы
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y==cl,N[0]], X[y==cl, N[1]],alpha=0.8, color=cmap(idx),marker=markers[idx], label=dict_[cl])
        
    plt.xlabel(f"{columns[0]}, см", fontsize=17)
    plt.ylabel(f"{columns[1]}, см", fontsize=17)
    plt.legend(prop={'size': 12})
    plt.show()


def plot_indent():
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot([-22, 0, 0, 22], [1, 1, 0, 0], label="$f(M_i) = [M_i<0]$")
    plt.xlabel("$M_i$", fontsize=17)
    plt.ylabel("$f(M_i)$", fontsize=17)
    plt.grid()
    plt.legend()
    plt.ylim([-1, 2])
    plt.xlim([-21, 21])
    plt.show()

def plot_indent_with_maj():
    plt.figure(figsize=(8, 4), dpi=100)
    x = np.linspace(-10, 8.0,100)
    y1 = (1 - x)**2
    y2 = np.array(list(map(lambda x: max(0, 1-x), x)))
    y3 = np.e**(-x)
    y4 = np.log(1 + y3)
    plt.plot([-22, 0, 0, 22], [1, 1, 0, 0], label="$[M_i<0]$")
    plt.plot(x,y1, label="$(1 - M_i)^2$")
    plt.plot(x,y2, label="$max(0,1 - M_i)$")
    plt.plot(x,y3, label="$e^{-M_i}$")
    plt.plot(x,y4, label="$ln(1 + e^{-M_i})$")
    plt.xlabel("$M_i$", fontsize=17)
    plt.ylabel("$g(M_i)$", fontsize=17)
    plt.grid()
    plt.legend()
    plt.ylim([-1, 7])
    plt.xlim([-7, 8])
    plt.show()


def plot_finaly_trained_model(X, kind_iris, theta):
    create_base_plot()
    teta_0, teta_1, teta_2 = theta
    plt.scatter(X[:,0][kind_iris=='iris setosa'], X[:,1][kind_iris=='iris setosa'], label='iris setosa', c='g')
    plt.scatter(X[:,0][kind_iris=='iris virginica'], X[:,1][kind_iris=='iris virginica'], label='iris virginica', c='b')
    plt.plot(X[:,0], (teta_0 +teta_1*X[:,0])/-teta_2)
    #plt.arrow(7,(teta_0 +teta_1*7)/-teta_2, teta_1/3,teta_2/3,head_width=0.2,width=0.01, label='$\Theta$')
    plt.legend(prop={'size': 12})
    plt.xlim([4, 8])
    plt.ylim([2, 4.5])
    plt.show()

def create_data(X):
    X_ones = np.ones(X.shape[0])
    return np.column_stack([X_ones, X])    

def poly(X,n=1):
    X1 = X.copy()
    for i in range(1,n):
        X1 = np.hstack((X1,X**(i+1)))
    return X1
        
    
def plot_decision_regions_softmax(X, y,columns, classifier , test_idx=None, resolution=0.02):
    # настроить генератор маркеров и палитру
    create_base_plot()
    dict_param = {"Длина чашелистника":0, "Ширина чашелистника":1,"Длина лепестка":2,"Ширина лепестка":3}
    dict_ = {0:"iris setosa",1:"iris versicolor",2:"iris virginica"}
    N = [dict_param[par] for par in columns]
    markers = ('s','x','o', '^','v')
    colors = ('red', 'blue', 'lightgreen','gray','суаn')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # вывести поверхность решения
    x1_min, x1_max = X[:,N[0]].min() - 1, X[:,N[0]].max() + 1
    x2_min, x2_max = X[:,N[1]].min() - 1, X[:,N[1]].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                           np.arange(x2_min,x2_max,resolution))
    X_field = create_data(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = classifier.predict(X_field)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # показать все образцы
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y==cl,N[0]], X[y==cl, N[1]],alpha=0.8, color=cmap(idx),marker=markers[idx], label=dict_[cl])
                
    plt.xlabel(f"{columns[0]}, см", fontsize=17)
    plt.ylabel(f"{columns[1]}, см", fontsize=17)
    plt.legend(prop={'size': 12})
    plt.show()

    
