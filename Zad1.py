import pandas as pd

data_frame = pd.read_csv('data.csv', header=None)

data_frame.columns('Długość Działki Kielicha [cm]',
                  'Szerokość Działki Kielicha [cm]',
                  'Długość Płatka [cm]',
                  'Szerokość Płatka [cm]',
                  'Gatunek')

cechy_numeryczne = data_frame.iloc[:, :-1]

cechy_zagregowane = cechy_numeryczne.agg(['min', 'max', 'mean', 'median', 'std', 'quantile']).transpose()

Q1 = cechy_zagregowane.quantile(0.25)
Q3 = cechy_zagregowane.quantile(0.75)
