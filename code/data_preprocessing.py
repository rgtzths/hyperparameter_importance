
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_abalone(source, destination):

    data = pd.read_csv(source)

    sex = data.pop('Sex')

    data['M'] = (sex == 'M')*1.0
    data['F'] = (sex == 'F')*1.0
    data['I'] = (sex == 'I')*1.0

    x_columns = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight','Viscera weight','Shell weight','M','F','I']
    y_column = ['Rings']

    X = data[x_columns]
    y = data[y_column]

    scalar= MinMaxScaler()

    data[x_columns] = scalar.fit_transform(data[x_columns])

    data[y_column]= scalar.fit_transform(data[y_column])

    data = data[x_columns + y_column]
    data.to_csv(destination, index=False)