import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from matplotlib.colors import ListedColormap
from ipywidgets import interact, IntSlider,  FloatSlider
import numpy as np
import pandas as pd
from sklearn import datasets


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


def plot_square(k_0,k_1):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x_minor_ticks = np.linspace(0, 1, k_0+1)
    y_minor_ticks = np.linspace(0, 1, k_1+1)
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_yticks(y_minor_ticks, minor=True)
    plt.xlabel("FPR", fontsize=17)
    plt.ylabel("TPR", fontsize=17)
    ax.grid(which='minor', alpha=1)
    plt.show()

def plot_roc_curve(fpr,tpr, k_0, k_1):
    fig = plt.figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    x_minor_ticks = np.linspace(0, 1, k_0+1)
    y_minor_ticks = np.linspace(0, 1, k_1+1)
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_yticks(y_minor_ticks, minor=True)
    plt.plot(fpr,tpr)
    plt.xlabel("FPR", fontsize=17)
    plt.ylabel("TPR", fontsize=17)
    ax.grid(which='minor', alpha=1)
    plt.title("ROC - CURVE")
    plt.show()


def plot_pr_curve(recall,precision):
    create_base_plot()
    plt.plot(recall,precision)
    plt.xlabel("recall", fontsize=17)
    plt.ylabel("precision", fontsize=17)
    plt.ylim([0, 1.1])
    plt.show()


def get_regress_data():
    N=40
    X = np.linspace(-1, 8, N)
    np.random.seed(0)
    return X.reshape(-1,1), 1/20*((X*(X+1)*(X-3)*(X-5)*(X-8) + 20*np.random.normal(size=(N,))))

def get_data_moon():
    M = 200
    X,y = datasets.make_moons(M)
    X1 = X.copy()
    y1 = y.copy()
    X1[y1 == 1,1] = X1[y1 == 1,1]-0.5
    X1[y1 == 0,1] = X1[y1 == 0,1]+0.5

    X = np.vstack((X,X1)) + np.random.randn(2*M,1)*0.1
    y = np.vstack((y.reshape(-1,1),y1.reshape(-1,1))).reshape(-1,)

    n = np.argmin(np.abs(X[:,0] - 1.5))
    y[n] = 0
    X[n,0] = 1
    X[n,1] = -0.75
    
    n = np.argmin(np.abs(X[:,0] - 0))
    y[n] = 0
    X[n,0] = 0.5
    X[n,1] = -1
    
    return X,y
    

def plot_regress_data(X,y):
    create_base_plot()
    plt.scatter(X,y)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()


def plot_regress_train_test_data(X_train, X_test, y_train, y_test):
    create_base_plot()
    plt.scatter(X_train,y_train, label="train")
    plt.scatter(X_test,y_test, label="test")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()

def plot_regress_data_and_model(X,y,lr):
    create_base_plot()
    plt.scatter(X[:,0],y)
    plt.plot(X[:,0], lr.intercept_ + X@lr.coef_)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()

def creat_polynomial_features(X, n):
    X_pol = X
    for i in range(1, n):
        X_pol = np.hstack((X_pol, X**i+1))
    return X_pol


def plot_train_test_mse(train_mse, test_mse):
    create_base_plot()
    plt.plot(range(1, len(train_mse)+1), train_mse,label="train MSE")
    plt.plot(range(1, len(train_mse)+1), test_mse, label="test MSE")
    plt.xlabel("Количество полиномиальных признаков", fontsize=17)
    plt.ylabel("MSE", fontsize=17)
    plt.legend()
    plt.ylim([0,200])
    plt.show()
    
def plot_data_moon(X,Y):
    create_base_plot()
    irises = set(Y)
    markers = ["o","x"]
    i = 0
    for iris in irises:
        plt.scatter(X[:,0][Y==iris], X[:,1][Y==iris], label=iris,marker = markers[i])
        i = i+1
    plt.xlabel(f"Признак 1", fontsize=17)
    plt.ylabel(f"Признак 2", fontsize=17)
    plt.legend(prop={'size': 12})
    plt.show() 
    
    
def plot_decision_regions_binary(X, y, classifier , test_idx=None, resolution=0.02,N = 1):
    # настроить генератор маркеров и палитру
    create_base_plot()
    markers = ('s','x','o', '^','v')
    colors = ('red', 'blue', 'lightgreen','gray','суаn')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # вывести поверхность решения
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                           np.arange(x2_min,x2_max,resolution))
    X_field = poly(np.array([xx1.ravel(), xx2.ravel()]).T,N)
    Z = classifier.predict(X_field)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # показать все образцы
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y==cl,0], X[y==cl, 1],alpha=0.8, color=cmap(idx),marker=markers[idx], label=f"Класс {cl}")
                
    plt.xlabel(f"Признак 1", fontsize=17)
    plt.ylabel(f"Признак 2", fontsize=17)
    plt.legend(prop={'size': 12})
    plt.show()    
    
def poly(X,n=1):
    X1 = X.copy()
    for i in range(1,n):
        X1 = np.hstack((X1,X**(i+1)))
    return X1    
