import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

columns = ['Długość działki kielicha',
              'Szerokość działki kielicha',
              'Długość płatka',
              'Szerokość płatka',
              'Gatunek']

df = pd.read_csv('data1.csv', header=None, names=columns)


# +-------------------------------------------------------------------------------------------------------------+
# |--------------------------------------------------Zadanie 1.1------------------------------------------------|
# +-------------------------------------------------------------------------------------------------------------+

mapa_garunkow = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['Gatunek'] = df['Gatunek'].map(mapa_garunkow)


ilosc_gatunkow = df['Gatunek'].value_counts()


procent_gatunkow = ((ilosc_gatunkow / len(df)) * 100).round(1)


zestawienie_gatunkow = pd.DataFrame({
    'Gatunek': ilosc_gatunkow.index,
    'Liczebność': ilosc_gatunkow.values,
    'Procent': procent_gatunkow})

podsumowanie_gatunkow = pd.DataFrame([{
    'Gatunek': 'Razem',
    'Liczebność': zestawienie_gatunkow['Liczebność'].sum(),
    'Procent': (zestawienie_gatunkow['Procent'].sum()).round(0)
}])

zestawienie_gatunkow = pd.concat([zestawienie_gatunkow, podsumowanie_gatunkow], ignore_index=True)

print(zestawienie_gatunkow.to_string(index=False, line_width=None, col_space=15, justify='center'))

# +-------------------------------------------------------------------------------------------------------------+
# |--------------------------------------------------Zadanie 1.2------------------------------------------------|
# +-------------------------------------------------------------------------------------------------------------+

columns2 = ['Długość działki kielicha',
              'Szerokość działki kielicha',
              'Długość płatka',
              'Szerokość płatka']

miary_rozkladu_list = []

for col in columns2:
    miary_rozkladu_list.append({
        'Cechy': col,
        'Maximum': df[col].max(),
        'Średnia': round(df[col].mean(), 2),
        'Odch. stand.': round(df[col].std(), 2),
        'Mediana': round(df[col].median(), 2),
        'Q1': round(df[col].quantile(0.25), 2),
        'Q3': round(df[col].quantile(0.75), 2),
        'Minimum': df[col].min(),
    })

miary_rozkladu = pd.DataFrame(miary_rozkladu_list) # Konwersja listy do obiektu DataFrame

print("\n")
print(miary_rozkladu.to_string(index=False, line_width=None, col_space=15, justify='center'))

# +-------------------------------------------------------------------------------------------------------------+
# |--------------------------------------------------Zadanie 2.1------------------------------------------------|
# +-------------------------------------------------------------------------------------------------------------+


# Histogram - Długość działki kielicha
przedzialy1 = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]

plt.hist(df['Długość działki kielicha'], bins=przedzialy1, color='blue', edgecolor='black')
plt.xticks(przedzialy1)
plt.title('Długość działki kielicha')
plt.xlabel('Wartości')
plt.ylabel('Liczebność')

plt.tight_layout()
plt.show()


# Histogram - Szerokość działki kielicha
przedzialy2 = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

plt.hist(df['Szerokość działki kielicha'], bins=przedzialy2, color='blue', edgecolor='black')
plt.xticks(przedzialy2)
plt.title('Szerokość działki kielicha')
plt.xlabel('Wartości')
plt.ylabel('Liczebność')

plt.tight_layout()
plt.show()


# Histogram - Długość płatka
przedzialy3 = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]

plt.hist(df['Długość płatka'], bins=przedzialy3, color='blue', edgecolor='black')
plt.xticks(przedzialy3)
plt.title('Długość płatka')
plt.xlabel('Wartości')
plt.ylabel('Liczebność')

plt.tight_layout()
plt.show()


# Histogram - Szerokość płatka
przedzialy4 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

plt.hist(df['Szerokość płatka'], bins=przedzialy4, color='blue', edgecolor='black')
plt.xticks(przedzialy4)
plt.title('Szerokość płatka')
plt.xlabel('Wartości')
plt.ylabel('Liczebność')

plt.tight_layout()
plt.show()

# +-------------------------------------------------------------------------------------------------------------+
# |--------------------------------------------------Zadanie 2.2------------------------------------------------|
# +-------------------------------------------------------------------------------------------------------------+

# Pudełkowy - Długość działki kielicha
dane_gatunek_0 = df[df['Gatunek'] == 'setosa']['Długość działki kielicha']
dane_gatunek_1 = df[df['Gatunek'] == 'versicolor']['Długość działki kielicha']
dane_gatunek_2 = df[df['Gatunek'] == 'virginica']['Długość działki kielicha']

plt.boxplot([dane_gatunek_0, dane_gatunek_1, dane_gatunek_2], tick_labels=['setosa', 'versicolor', 'virginica'])

plt.title('Długość działki kielicha')
plt.ylabel('Długość (cm)')

plt.tight_layout()
plt.show()

# Pudełkowy - Szerokość działki kielicha
dane_gatunek_0 = df[df['Gatunek'] == 'setosa']['Szerokość działki kielicha']
dane_gatunek_1 = df[df['Gatunek'] == 'versicolor']['Szerokość działki kielicha']
dane_gatunek_2 = df[df['Gatunek'] == 'virginica']['Szerokość działki kielicha']

plt.boxplot([dane_gatunek_0, dane_gatunek_1, dane_gatunek_2], tick_labels=['setosa', 'versicolor', 'virginica'])

plt.title('Szerokość działki kielicha')
plt.ylabel('Szerokość (cm)')

plt.tight_layout()
plt.show()

# Pudełkowy - Długość płatka
dane_gatunek_0 = df[df['Gatunek'] == 'setosa']['Długość płatka']
dane_gatunek_1 = df[df['Gatunek'] == 'versicolor']['Długość płatka']
dane_gatunek_2 = df[df['Gatunek'] == 'virginica']['Długość płatka']

plt.boxplot([dane_gatunek_0, dane_gatunek_1, dane_gatunek_2], tick_labels=['setosa', 'versicolor', 'virginica'])

plt.title('Długość płatka')
plt.ylabel('Długość (cm)')

plt.tight_layout()
plt.show()

# Pudełkowy - Szerokość płatka
dane_gatunek_0 = df[df['Gatunek'] == 'setosa']['Szerokość płatka']
dane_gatunek_1 = df[df['Gatunek'] == 'versicolor']['Szerokość płatka']
dane_gatunek_2 = df[df['Gatunek'] == 'virginica']['Szerokość płatka']

plt.boxplot([dane_gatunek_0, dane_gatunek_1, dane_gatunek_2], tick_labels=['setosa', 'versicolor', 'virginica'])

plt.title('Szerokość płatka')
plt.ylabel('Szerokość (cm)')

plt.tight_layout()
plt.show()

# +-------------------------------------------------------------------------------------------------------------+
# |--------------------------------------------------Zadanie 3.1------------------------------------------------|
# +-------------------------------------------------------------------------------------------------------------+


# Szerokość działki kielicha - Długość działki kielicha
korelacja = (df['Szerokość działki kielicha'].corr(df['Długość działki kielicha'])).round(2)

slope, intercept = np.polyfit(df['Długość działki kielicha'], df['Szerokość działki kielicha'], 1) # Współczyniki równania równania regresji liniowej

sns.lmplot(x='Długość działki kielicha', y='Szerokość działki kielicha', data=df, ci=None, line_kws={'color': 'red'}, scatter_kws={'s': 50,'color': 'blue'})

plt.title(f'r = {korelacja}; y = {slope:.1f}x + {intercept:.1f}')
plt.xlabel('Długość działki kielicha (cm)')
plt.ylabel('Szerokość działki kielicha (cm)')

plt.tight_layout()
plt.show()

# Długość płatka - Długość działki kielicha
korelacja = (df['Długość płatka'].corr(df['Długość działki kielicha'])).round(2)

slope, intercept = np.polyfit(df['Długość działki kielicha'], df['Długość płatka'], 1) # Współczyniki równania równania regresji liniowej

sns.lmplot(x='Długość działki kielicha', y='Długość płatka', data=df, ci=None, line_kws={'color': 'red'}, scatter_kws={'s': 50,'color': 'blue'})

plt.title(f'r = {korelacja}; y = {slope:.1f}x + {intercept:.1f}')
plt.xlabel('Długość działki kielicha (cm)')
plt.ylabel('Długość płatka (cm)')

plt.tight_layout()
plt.show()

# Szerokość płatka - Długość działki kielicha
korelacja = (df['Szerokość płatka'].corr(df['Długość działki kielicha'])).round(2)

slope, intercept = np.polyfit(df['Długość działki kielicha'], df['Szerokość płatka'], 1) # Współczyniki równania równania regresji liniowej

sns.lmplot(x='Długość działki kielicha', y='Szerokość płatka', data=df, ci=None, line_kws={'color': 'red'}, scatter_kws={'s': 50,'color': 'blue'})

plt.title(f'r = {korelacja}; y = {slope:.1f}x + {intercept:.1f}')
plt.xlabel('Długość działki kielicha (cm)')
plt.ylabel('Szerokość płatka (cm)')

plt.tight_layout()
plt.show()

# Długość płatka - Szerokość działki kielicha
korelacja = (df['Długość płatka'].corr(df['Szerokość działki kielicha'])).round(2)

slope, intercept = np.polyfit(df['Szerokość działki kielicha'], df['Długość płatka'], 1) # Współczyniki równania równania regresji liniowej

sns.lmplot(x='Szerokość działki kielicha', y='Długość płatka', data=df, ci=None, line_kws={'color': 'red'}, scatter_kws={'s': 50,'color': 'blue'})

plt.title(f'r = {korelacja}; y = {slope:.1f}x + {intercept:.1f}')
plt.xlabel('Szerokość działki kielicha (cm)')
plt.ylabel('Długość płatka (cm)')

plt.tight_layout()
plt.show()

# Szerokość płatka - Szerokość działki kielicha
korelacja = (df['Szerokość płatka'].corr(df['Szerokość działki kielicha'])).round(2)

slope, intercept = np.polyfit(df['Szerokość działki kielicha'], df['Szerokość płatka'], 1) # Współczyniki równania równania regresji liniowej

sns.lmplot(x='Szerokość działki kielicha', y='Szerokość płatka', data=df, ci=None, line_kws={'color': 'red'}, scatter_kws={'s': 50,'color': 'blue'})

plt.title(f'r = {korelacja}; y = {slope:.1f}x + {intercept:.1f}')
plt.xlabel('Szerokość działki kielicha (cm)')
plt.ylabel('Szerokość płatka (cm)')

plt.tight_layout()
plt.show()

# Szerokość płatka - Długość płatka
korelacja = (df['Szerokość płatka'].corr(df['Długość płatka'])).round(2)

slope, intercept = np.polyfit(df['Długość płatka'], df['Szerokość płatka'], 1) # Współczyniki równania równania regresji liniowej

sns.lmplot(x='Długość płatka', y='Szerokość płatka', data=df, ci=None, line_kws={'color': 'red'}, scatter_kws={'s': 50,'color': 'blue'})

plt.title(f'r = {korelacja}; y = {slope:.1f}x + {intercept:.1f}')
plt.xlabel('Długość płatka (cm)')
plt.ylabel('Szerokość płatka (cm)')

plt.tight_layout()
plt.show()
