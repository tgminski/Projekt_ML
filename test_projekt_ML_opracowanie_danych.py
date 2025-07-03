''' test_projekt_ML_opracowanie_danych '''


import numpy as np
import pandas as pd
import seaborn as sns

DATA_FILE_RAW = 'cars.csv'


def main():

    print('-------------------------------------------')
    print('Wczytywanie danych z pliku "cars.csv"')
    df_raw = load_data_from_csv(DATA_FILE_RAW)

    # pierwsze spojrzenie na dane
    print(df_raw.info())
    print(df_raw.head())
    print(df_raw.describe())
    # dane kategoryczne: origin
    # na podstawie describe: zakresy danych w kolumnach mieszczą się w logicznych wartościach

    df_raw = sprawdzenie_poprawności_danych(df_raw)   # na podsrawie informacji o danych
    
    df_no_nan = sprawdzenie_nan(df_raw)  # sprawdzenie obecności NaN w danych

    columns = df_no_nan.columns.tolist()
    columns_numeric = [columns[0]] + columns[2:7]  # kolumny numeryczne: 'mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']
    columns_categorical = [columns[1]] + [columns[7]]
    #print(columns)
    print('---- Kolumny z danymi numerycznymi:')
    print(columns_numeric)
    print('---- Kolumny z danymi kategorycznymi:')
    print(columns_categorical)

    df_dummy = zamiana_danych_kategorycznych_na_liczbowe(df_no_nan, columns_numeric, columns_categorical)
    #print(df_dummy.info())
    df_names = df_dummy[['name']]   # zapamiętanie kolumny 'name' przed usunięciem
    #print(df_names.head())
    df_dummy = df_dummy.drop(columns=['name'])  # usunięcie kolumny 'name' z danych

    columns = df_dummy.columns.tolist()
    columns_numeric = columns[:6]  # kolumny numeryczne: 'mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']
    columns_categorical = columns[6:]
    #print(columns)
    print('---- Kolumny z danymi numerycznymi:')
    print(columns_numeric)
    print('---- Kolumny z danymi kategorycznymi:')
    print(columns_categorical)

    df_outliers = sprawdzenie_outliers(df_dummy,columns_numeric,columns_categorical)

    print('-------------------------------------------')
    print('---- Usuwanie outliers z danych')
    print('-------------------------------------------')

    df_dummy = pd.concat([df_names, df_dummy], axis=1)  # dodanie kolumny 'name' do danych
    df_dummy_no_outl = df_dummy[~df_outliers['is_outlier']].copy()  # usunięcie outliers z danych

    print(df_dummy_no_outl.info())
    print(df_dummy_no_outl.iloc[:,7:].sum())

    save_df_data_to_csv(df_dummy_no_outl, 'cars_dummy_no_outliers.csv')  # zapisanie danych bez outliers do pliku

    print('-------------------------------------------')
    print('---- Dane zapisano do pliku "cars_dummy_no_outliers.csv"')
    print('-------------------------------------------')

    df_dummy_all_no_outl = dane_numeryczne_z_kolumny_names(df_dummy_no_outl)

    save_df_data_to_csv(df_dummy_all_no_outl, 'cars_dummy_all_no_outliers.csv')  # zapisanie danych z kolumny 'name' do pliku
    print('-------------------------------------------')
    print('---- Dane zapisano do pliku "cars_dummy_all_no_outliers.csv"')
    print('-------------------------------------------')

    return []


def sprawdzenie_poprawności_danych(df_raw):

    # usunięcie wartości z cylinders = 3
    df_raw = df_raw[df_raw['cylinders']!=3].copy()

    return df_raw

def sprawdzenie_nan(df_raw):

    print('-------------------------------------------')
    print('Sprawdzanie obecności NaN w danych')
    print('-------------------------------------------')

    print("---- Checking for NaN values in the DataFrame:")
    print(df_raw.isna().sum())

    if df_raw.isna().sum().sum() > 0:
        print("---- There are NaN values in the DataFrame.")
        df_raw = df_raw.dropna()
        print("---- NaN values have been removed.")
    else:
        print("---- No NaN values found.")

    return df_raw

def zamiana_danych_kategorycznych_na_liczbowe(df, columns_numeric, columns_categorical):
    print('-------------------------------------------')
    print('Zamiana danych kategorycznych na liczbowe.')
    print('-------------------------------------------')
    
    # zamiana kolumny 'origin' na dane liczbowe (dummy)
    df_dummy = df.copy()
    for col in columns_categorical:
        df_dummy_col = pd.get_dummies(df[col], prefix=col) #, drop_first=True)
        df_dummy = pd.concat([df_dummy, df_dummy_col], axis=1)
        df_dummy = df_dummy.drop(columns=[col])  

    return df_dummy

def sprawdzenie_outliers(df_dummy,columns_numeric,columns_categorical):
    print('-------------------------------------------')
    print('Sprawdzanie obecności outliers w danych')
    print('-------------------------------------------')

    # wykresy rozrzutu dla wszystkich kolumn numerycznych
    #sns.pairplot(df_dummy)

    
    

    # sprawdzanie danych kategorycznych
    print('---- Ilości danych kategorycznych:')
    print(df_dummy[columns_categorical].sum())
    print('---- % zawartości kategorii w całym zbiorze')
    print(df_dummy[columns_categorical].mean().round(4) * 100)
    print('---- Kategorie w kolumnie "origin" są reprezentatywne.')

    #df_outliers = df_dummy[columns_numeric].copy()
    df_outliers = pd.DataFrame()
    print(df_outliers.info())
    for col_num in columns_numeric:
        for col_cat in columns_categorical:
            #print(col_num, col_cat)
            filter = df_dummy[col_cat] == 1 
            Q1 = df_dummy.loc[filter, col_num].quantile(0.25)
            Q3 = df_dummy.loc[filter, col_num].quantile(0.75)
            IQR = Q3 - Q1
            #print(Q1, Q3, IQR)
            outliers = ((df_dummy[col_num] < (Q1 - 1.5 * IQR )) & filter  ) | ((df_dummy[col_num] > (Q3 + 1.5 * IQR)) & filter)
            #print(outliers)
            #df_outliers[col_num] = pd.Series(dtype='bool')
            col_name = col_num+ '_' + col_cat
            df_outliers[col_name] = outliers
            if outliers.any():
                print(f"------ Wykryto outlier w kolumnie '{col_num}' dla kategorii '{col_cat}' w ilości '{outliers.sum()}'.")
                
                print(df_dummy.loc[outliers, col_num])
            else:
                print(f"------ Brak outlier w kolumnie '{col_num}' dla kategorii '{col_cat}'.")

            print(len(outliers), outliers.sum())
            print(len(filter), sum(filter), Q1, Q3, IQR)
            #df_outliers
    
    

    df_outliers['is_outlier'] = df_outliers.any(axis=1)
    print('---- Łącznie wykryto ', df_outliers['is_outlier'].sum(), 'outliers.')

    print(df_outliers.sum())
    #print(df_outliers.head())

    print('---- lista wykrytych outliers')
    print(df_dummy[df_outliers['is_outlier']])

    return df_outliers

def dane_numeryczne_z_kolumny_names(df_dummy_no_outl):

    print('-------------------------------------------')
    print('Dane numeryczne z kolumny names')
    print('-------------------------------------------')

    feature_str_list = []
    for i in df_dummy_no_outl['name'].tolist():
        #print(type(i))
        #print(i)
        i_split = i.split(' ')
        #print(i_split)

        #assert 0, 'Przerwano działanie'

        for i1 in i_split:
            feature_str_list = feature_str_list + [i1]

    #print(len(feature_str_list))
    #print(feature_str_list)
        
    feature_set = set(feature_str_list)
    #print(len(feature_set))
    #print(feature_set)
    df_dummy_all_no_outl = df_dummy_no_outl.copy()
    names = df_dummy_no_outl['name'].to_numpy(dtype=str).copy()
    for feature in feature_set:
        col_name = 'name_' + feature
        #for name in np.nditer(names):
        df_dummy_all_no_outl[col_name] = np.strings.count(names, feature, start=0, end=None).reshape((-1,))
            
    print(df_dummy_all_no_outl.info())
    print(df_dummy_all_no_outl.head())

        
    return df_dummy_all_no_outl

def load_data_from_csv(file):
    data = pd.read_csv(file)
    return data

def save_df_data_to_csv(data, file):
    data.to_csv(file, index=False)
    return []

if __name__ == '__main__':
    main()