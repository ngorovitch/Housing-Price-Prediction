# HOUSING PRICE PREDICTION WITH FULL LASSO (PYTHON)
The data set for this project is available here : https://www.kaggle.com/c/house-prices-advanced-regression-techniques

## Data tidying
1. Picking the 3 most correlated features to the sale price and removal of outliers. In this
case, if SalePrice <3000000 and GrLivArea >4000. This is done because since they are
highly correlated to the sale price, having those outliers might decrease the accuracy of
our prediction.
2. Conversion of non-numerical variables into dummie variables by using
pandas.get_dummies(). It is done to change strings in our dataset into numerical
dummie values to facilitate the predictions made by our model
3. Replacing NA by the mean of other elements in a column
## Feature selection
1. Creation of a train correlation heatmap to figure out the relation between the various
feature
2. Checking for multicolinearity of all numerical features except SalePrice. We will keep the
combination of highly correlated features. We consider highly correlated features as
two features having a correlation score greater than 0.7 and smaller than -0.7.
3. Removing '1stFlrSF', 'GarageArea' and 'TotRmsAbvGrd' as they are less correlated to the
Saleprice than the other member of the couple; but keeping 'GarageYrBlt' and
'YearBuilt' because they show weak correlation with Saleprice
## Feature Engineering
1. Log transformation applied to SalePrice in order to have normally distributed dependent
variable
2. Creation of a new variable called TotalSF representing the total surface area which is the
combination ‘TotalBsmtSF' and '2ndFlrSF'.
3. multiplying predictions too close to the upper bound of all predictions by 0.99 trying to
switch them down; multiplying predictions too close to the lower bound of all
predictions by 1.05 trying to switch them up
## Model Selection
1. Lasso with alpha=0.0004 (Kaggle Score: 0.11634)
