import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv('./data/train.csv')
df.head()

df.shape
df.isnull().values.any()
df.corr()

def plot_corr(df, size=20):
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot

    Displays:
        matrix of correlation between columns.  Blue-cyan-yellow-red-darkred => less to more correlated
                                                0 ------------------>  1
                                                Expect a darkred line running from top left to bottom right
    """

    corr = df.corr()    # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)   # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks

plot_corr(df)
df.dtypes

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

y=df.E
X=df.drop(['E','Ob','T6'],axis=1)

train_X,test_X,train_y,test_y=train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_model = XGBRegressor(n_estimators=50000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=230, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

Test_df=pd.read_csv('./data/test.csv')
Test_X=Test_df.drop(['Ob','T6'],axis=1)
predictions = my_model.predict(Test_X.as_matrix())

print(predictions)

my_submission = pd.DataFrame({'Observation': Test_df.Ob, 'Energy': predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False,columns=['Observation','Energy'])
