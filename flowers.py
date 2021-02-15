import pandas as pd
import matplotlib.pyplot as plt

# wczytywanie danych
iris = pd.read_csv(r"C:\Users\danon\OneDrive\Pulpit\MACHINE LEARNING\iris1.csv")

# rysowanie wykresów dla płatków (petal)                   
xp_min = iris['petal.length'].min()
xp_max = iris['petal.length'].max()
yp_min = iris['petal.width'].min() 
yp_max = iris['petal.width'].max() 

colors = {'Setosa':'red', 'Versicolor':'blue', 'Virginica':'green'}

fig, ax = plt.subplots(figsize=(11, 11))

for key, group in iris.groupby(by = 'variety'):
    plt.scatter(group['petal.length'], group['petal.width'], 
                c = colors[key], label = key)
    
plt.legend()
plt.grid()
plt.xlabel('petal.length')
plt.ylabel('petal.width')
plt.xlim(xp_min - 0.5, xp_max + 0.5)
plt.ylim(yp_min - 0.5, yp_max + 0.5)
ax.set_title("IRIS DATASET CATEGORIZED")

# rysowanie wykresów dla sepal
xs_min = iris['sepal.length'].min()
xs_max = iris['sepal.length'].max()
ys_min = iris['sepal.width'].min()
ys_max = iris['sepal.width'].max()



fig,ax = plt.subplots(figsize=(11,11))

for key,group in iris.groupby(by='variety'):
    plt.scatter(group['sepal.length'],group['sepal.width'], 
                c = colors[key], label = key)
    
plt.legend()
plt.grid()
plt.xlabel('sepal.length')
plt.ylabel('sepal.width')
plt.xlim(xs_min - 0.5, xs_max + 0.5 )
plt.ylim(ys_min - 0.5, ys_max + 0.5)
ax.set_title("IRIS DATASET CATEGORIZED")

plt.show()


fig.ax = plt.subplots(2,2,figsize = (11,11))

plt_position = 1

feature_x = 'petal.width'  #zaleznosc tej zmiennej a pozostałymi cechami

for feature_y in iris.columns[:4]:  # /dla kazdej cechy opisujacej kwiaty/
    plt.subplot(2,2,plt_position)
    for variety, color in colors.items():
        plt.scatter(iris.loc[iris['variety'] == variety, feature_x],
                    iris.loc[iris['variety'] == variety, feature_y],
                    label = variety,
                    alpha = 0.45, # transparency
                    color = color)

    # opisujemy wykres
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt_position += 1

plt.show()


pd.plotting.scatter_matrix(iris, figsize=(11, 11), 
                          color = iris['variety'].apply(lambda x: colors[x]));
plt.show()

# ... a tutaj podobny wykres generowany przez funkcję pairplot z modułu seaborn
import seaborn as sns
sns.set()
sns.pairplot(iris, hue="variety")



# modele predykcyjne
from sklearn.linear_model import LinearRegression

iris = pd.read_csv(r"C:\Users\danon\OneDrive\Pulpit\MACHINE LEARNING\iris1.csv")

X_train = iris.iloc[:,:4]
y_train = iris.loc[:,'variety']

categories = {'Setosa':1, 'Versicolor':2, 'Virginica':3}
y = y_train.apply(lambda x: categories[x])

# tworzymy model
model = LinearRegression()
model.fit(X_train,y)
print(model.score(X_train,y))

# przykład działania modelu

flower_1 = [3,2.1,1.4,0.3]
flower_2 = [6, 4.1, 3.3, 2.7]
flower_3 = [5.5, 2, 1,0.1]

flowers = [flower_1,flower_2,flower_3]

variety_predict = model.predict(flowers)
print(variety_predict)

for i,j in zip(flowers, variety_predict):
    if round(j) == 1:
        print('{} is {}'.format(i,'Setosa'))
    elif round(j) == 2:
        print('{} is {}'.format(i,'Versicolor'))
    elif round(j) == 3:
        print('{} is {}'.format(i,'Virginica'))
    else:
        print('Unknown flower')