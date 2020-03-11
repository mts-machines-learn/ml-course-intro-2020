import matplotlib.pyplot as plt
from sklearn import datasets
from ipywidgets import interact, IntSlider,  FloatSlider
import numpy as np
import pandas as pd


def get_data(irises = ["iris setosa","iris virginica"]):
    iris_dict = {0:"iris setosa",1:"iris versicolor",2:"iris virginica"}
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = np.array([iris_dict[iris] for iris in iris.target])
    return X[np.isin(y,irises)], y[np.isin(y,irises)]


def print_data(X,Y):
    df = pd.DataFrame(np.hstack((X, Y.reshape(-1,1))), columns=['Длина чашелистника', 'Ширина чашелистника', 'Вид Ириса'])
    print(df)


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


def plot_data(X,Y):
    create_base_plot()
    plt.scatter(X[:,0][Y=='iris setosa'], X[:,1][Y=='iris setosa'], label='iris setosa', c='g')
    plt.scatter(X[:,0][Y=='iris virginica'], X[:,1][Y=='iris virginica'], label='iris virginica', c='b')
    plt.xlabel("Длина чашелистника, см", fontsize=17)
    plt.ylabel("Ширина чашелистника, см", fontsize=17)
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


def plot_sigmoid():
    plt.figure(figsize=(8, 4), dpi=100)
    x = np.linspace(-7,7,100)
    y = 1/(1+np.e**-x)
    plt.plot(x,y, label="$\sigma(X\Theta)$")
    plt.xlabel("$X\Theta$", fontsize=17)
    plt.ylabel("$\sigma$", fontsize=17)
    plt.grid()
    plt.legend()
    plt.xlim([-7, 7])
    plt.show()



def plot_div_mse_ce():
    plt.figure(figsize=(8, 4), dpi=100)
    x = np.linspace(0,1,50)
    y1 = np.abs(-2*(1-x))
    y2 = np.abs(-(1+np.e**-x))
    plt.plot(x,y1, label="|div MSE|/DS")
    plt.plot(x,y2, label="|div cross entropy|/DS")
    plt.grid()
    plt.legend()
    plt.show()

def plot_roc():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x_minor_ticks = np.linspace(0, 1, 5)
    y_minor_ticks = np.linspace(0, 1, 4)
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_yticks(y_minor_ticks, minor=True)
    ax.grid(which='minor', alpha=1)

