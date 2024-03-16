from imports import np, pd, os

path = os.path.join("../data","months.csv")

def read_data(path = path):
    """
    Read the data from the csv file and return a pandas dataframe

    Args:
        path: path to the csv file
    
    Returns:
        data: pandas dataframe
    """

    column_names = ['Greg_Year', 'Month', 'Dec_Year', 'N_total_sunspots_smoothed', 'Montly_mean_sunspot_number_std', 'N_obs', 'Marker']
    data = pd.read_csv(path, names=column_names, sep=";")
    data = data[data["N_total_sunspots_smoothed"] >= 0]

    return data