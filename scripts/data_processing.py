from imports import np, pd, os, rfft, rfftfreq, irfft, tf, plt

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




def filter_data(df, x, y, treshold=0.1):
    """
    Filter the data to remove high frequencies that are fluctuating too much

    Args:
        df: the pandas dataframe with all features
        treshold: the treshold to remove the high frequencies
        x: the column name of the x axis
        y: the column name of the y axis
    
    Returns:
        filtered: the filtered data
    """

    df = df[[x, y]]

    ft = rfft(df[y].to_numpy())
    frequencies = rfftfreq(len(df[y]), d=df[x].diff().mean())

    ft[frequencies > treshold] = 0
    filtered = irfft(ft)

    return filtered



def split_data(df, split_ratio = 0.7):
    """
    Split the data into a training and a test set

    Args:
        df: filtered pandas dataframe
        split_time
    
    Returns:
        train: the training set
        test: the test set
    """
    n = len(df)
    split_time = int(n*split_ratio)
    train = df[:split_time]
    test = df[split_time:]

    return train, test



def standardize_data(train, test):
    """
    Standardize the data

    Args:
        train: the training set
        test: the test set
    
    Returns:
        df: the standardized dataframe
    """
    temp_train = train.copy()
    temp_test = test.copy()
    temp_train = temp_train['N_total_sunspots_smoothed']
    temp_test = temp_test['N_total_sunspots_smoothed']

    mean = temp_train.mean()
    std = temp_train.std()

    train['N_total_sunspots_smoothed'] = (train['N_total_sunspots_smoothed'] - mean) / std
    test['N_total_sunspots_smoothed'] = (test['N_total_sunspots_smoothed'] - mean) / std

    return train, test



class window_generator():
    """
    Class to generate the windows for the model
    """
    def __init__(self, input_width, label_width, shift, train_df, test_df, label_columns=None):
        """
        Initialize the window generator

        Args:
            input_width: the width of the input
            label_width: the width of the label
            shift: the shift
            train_df: the training set
            test_df: the test set
            label_columns: the columns of the labels
        """
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.train_df = train_df
        self.test_df = test_df
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    
def split_window(self, features):
    """
    Split the window into inputs and labels

    Args:
        features: the features
    
    Returns:
        inputs: the inputs
        labels: the labels
    """
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


def make_dataset(self, data):
    """
    Maps the data to the split window function

    Args:
        data: the data

    Returns:
        ds: the mapped dataset
    """
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)
    ds = ds.map(self.split_window)

    return ds



@property
def train(self):
    """
    Get and cache the training set
    """
    return self.make_dataset(self.train_df)

@property
def test(self):
    """
    Get and cache the test set
    """
    return self.make_dataset(self.test_df)

@property
def example(self):
    """
    Get and cache an example batch of `inputs, labels` for plotting
    """
    result = getattr(self, '_example', None)
    if result is None:
      result = next(iter(self.train))
      self._example = result
    return result

def plot(self, model=None, plot_col='N_total_sunspots_smoothed', max_subplots=3):
    """
    Plot the data with the corresponding labels and predictions when a model is given

    Args:
        (model: the model)
        plot_col: the column to plot
        max_subplots: the maximum number of subplots
    """
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(3, 1, n+1)
        plt.ylabel(f'Number of sunspots [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index], 'b-', label='Inputs', zorder=-10)
        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index
        if label_col_index is None:
            continue
        plt.plot(self.label_indices, labels[n, :, label_col_index], 'r-', label='Labels', zorder=-10)
        if model is not None:
            predictions = model(inputs)
            plt.plot(self.label_indices, predictions[n, :, label_col_index], 'g-', label='Predictions', zorder=-10)
        if n == 0:
            plt.legend()
    plt.xlabel('Year')

#following code adds the functions to the class
window_generator.split_window = split_window
window_generator.make_dataset = make_dataset
window_generator.train = train
window_generator.test = test
window_generator.plot = plot
window_generator.example = example

MAX_EPOCHS = 500

def compile_and_fit(model, window, lr, momentum, patience=3):
    """
    Compile and fit the model

    Args:
        model: the model
        window: the window
        lr: the learning rate
        momentum: the momentum
        patience: the patience for the early stopping
    
    Returns:
        history: fitted model
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                      patience=patience,
                                                      mode='auto')

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                  optimizer=tf.keras.optimizers.Adam(beta_1=momentum, learning_rate=lr),
                  metrics=["mse", "mae"])

    history = model.fit(window.train, epochs=MAX_EPOCHS, callbacks=[early_stopping])
    return history




def dense_model(outer_steps):
    """
    Creates a dense model

    Args:
        outer_steps: the number of outer steps

    Returns:
        multi_dense_model: the dense model
    """
    multi_dense_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        
        tf.keras.layers.Dense(1024, activation='relu'),
        
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(outer_steps * 2,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([outer_steps, 2])
    ])
    return multi_dense_model

def linear_model(outer_steps):
    """
    Creates a linear model

    Args:
        outer_steps: the number of outer steps

    Returns:
        multi_linear_model: the linear model
    """
    multi_linear_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(outer_steps * 2,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([outer_steps, 2])
    ])
    return multi_linear_model