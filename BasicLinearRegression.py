# import pandas as pd
# import tensorflow as tf
#
#
# dfTrain = pd.read_csv('Gemini_BTCUSD_1h.csv',parse_dates=['date']) # Training Data
# dfEval = pd.read_csv('Gemini_BTCUSD_d.csv',parse_dates=['date']) # Testing Data
# y_train = dfTrain.pop('open')
# y_eval = dfEval.pop('open')
# # unix,date,symbol,open,high,low,close,Volume BTC,Volume USD
# dfEval.pop('symbol')
# dfTrain.pop('symbol')
# dfTrain['high'] = dfTrain['high'].astype(float)
# dfTrain['low'] = dfTrain['low'].astype(float)
# # dfTrain['open'] = dfTrain['open'].astype(float)
# dfTrain['close'] = dfTrain['close'].astype(float)
# dfTrain['Volume BTC'] = dfTrain['Volume BTC'].astype(float)
# dfTrain['Volume USD'] = dfTrain['Volume USD'].astype(float)
# # unix,date,symbol,open,high,low,close,Volume BTC,Volume USD
# dfEval['high'] = dfEval['high'].astype(float)
# dfEval['low'] = dfEval['low'].astype(float)
# # dfEval['open'] = dfEval['open'].astype(float)
# dfEval['close'] = dfEval['close'].astype(float)
# dfEval['Volume BTC'] = dfEval['Volume BTC'].astype(float)
# dfEval['Volume USD'] = dfEval['Volume USD'].astype(float)
#
# dfTrain['date'].to_numpy()
# dfEval['date'].to_numpy()
# print("Training Data using BTC 1hr historical data.")
# print(dfTrain.head())
# print("Evaluation Data using BTC day historical data.")
# print(dfEval.head())
#
#
# print(dfTrain.columns.all())
#
# CATAGORICAL_COLUMNS = []
#
# NUMERIC_COLUMNS = ['unix','date',"high","low","close","Volume BTC","Volume USD"]
# feature_columns = []
# print("Catagorical Columns: ",len(CATAGORICAL_COLUMNS), "\nNumeric Columns: ", len(NUMERIC_COLUMNS),
#       '\n Feature Columns: ', len(feature_columns))
# for feature_name in CATAGORICAL_COLUMNS:
#     vocabulary = dfTrain[feature_name].unique()
#     feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
# for feature_name in NUMERIC_COLUMNS:
#   feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
#
# print("Catagorical Columns: ",len(CATAGORICAL_COLUMNS), "\nNumeric Columns: ", len(NUMERIC_COLUMNS),
#       '\n Feature Columns: ', len(feature_columns))
#
# print("Shape of Training Data: ", dfTrain.shape)
# print("Shape of Evaluation Data: ", dfEval.shape)



import pandas as pd
import tensorflow as tf

dfTrain = pd.read_csv('Gemini_BTCUSD_1h.csv',parse_dates=['date']) # Training Data
dfEval = pd.read_csv('Gemini_BTCUSD_d.csv',parse_dates=['date']) # Testing Data
y_train = dfTrain.pop('open')
y_eval = dfEval.pop('open')
dfEval.pop('symbol')
dfTrain.pop('symbol')
dfTrain['high'] = dfTrain['high'].astype(float)
dfTrain['low'] = dfTrain['low'].astype(float)
dfTrain['close'] = dfTrain['close'].astype(float)
dfTrain['Volume BTC'] = dfTrain['Volume BTC'].astype(float)
dfTrain['Volume USD'] = dfTrain['Volume USD'].astype(float)
dfEval['high'] = dfEval['high'].astype(float)
dfEval['low'] = dfEval['low'].astype(float)
dfEval['close'] = dfEval['close'].astype(float)
dfEval['Volume BTC'] = dfEval['Volume BTC'].astype(float)
dfEval['Volume USD'] = dfEval['Volume USD'].astype(float)

# preprocess date column to extract year, month, and day
dfTrain['date'] = (pd.DatetimeIndex(dfTrain['date']).year + pd.DatetimeIndex(dfTrain['date']).month + pd.DatetimeIndex(dfTrain['date']).day
+ pd.DatetimeIndex(dfTrain['date']).hour)

dfEval['date'] = (pd.DatetimeIndex(dfEval['date']).year + pd.DatetimeIndex(dfEval['date']).month + pd.DatetimeIndex(dfEval['date']).day
+ pd.DatetimeIndex(dfEval['date']).hour)


feature_columns = []
CATAGORICAL_COLUMNS = []
# add date as a numeric feature
NUMERIC_COLUMNS = ['unix', 'high', 'low', 'close', 'Volume BTC', 'Volume USD']
date_feature = tf.feature_column.numeric_column('date')
feature_columns.append(date_feature)

# add year, month, and day as numeric features
NUMERIC_COLUMNS += ['year', 'month', 'day']
feature_columns += [    tf.feature_column.numeric_column('year'),    tf.feature_column.numeric_column('month'),    tf.feature_column.numeric_column('day')]

print("Training Data using BTC 1hr historical data.")
print(dfTrain.head())
print("Evaluation Data using BTC day historical data.")
print(dfEval.head())

print("Catagorical Columns: ",len(CATAGORICAL_COLUMNS), "\nNumeric Columns: ", len(NUMERIC_COLUMNS),
      '\nFeature Columns: ', len(feature_columns))
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=8):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dfTrain, y_train)


# here we will call the input_function that was returned to us to get a dataset object we can feed to the model


eval_input_fn = make_input_fn(dfEval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimator by passing the feature columns we created earlier

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

print(result['accuracy'])  # the result variable is simply a dict of stats about our model

pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
#
probs.plot(kind='hist', bins=20, title='predicted probabilities')



