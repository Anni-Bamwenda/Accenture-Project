# Accenture-project-22-

## Business Consideration

Proposing optimal locations is beneficial to Accenture because it offers exposure to new clients, maintains long-lasting relationships with existing clients, and maximizes their profits while minimizing loss in expansion

## Description

Aim of the project is to predict 3 optimal city locations where Accenture should open new Offices. Data sources used are Accenture office locations data(from Accenture), US cities economic data(from Census) and Fortune 500 Corporate data(from Census). To implement the project, we will learn different approaches to cleaning and transforming data to prepare it for our Logistic Regression model. 

Throughout this project, we will create the following code/system:
- A new dataframe by using aggregated columns/features that have been cleaned and gone through some manipulation.
- Logistic Regression model to predict future city locations.
- Evaluate the performance of our model using accuracy score and log loss.



## Software and Libraries
* python  
* NumPy
* pandas
* seaborn
* matplotlib
* scikit-learn


## General Workflow
The general workflow used to create the model is:

### Data Preprocessing:
These preprocessing steps are integral to model performance later on as they improve the quality and interpretability of the dataset. 
We will apply drop unnecessary and redundant data from the datasets then create a new dataset. 

* Step 01: Dropping repeated columns and unnecessary features(some features from the census data will not help us to reach our business goal, ex: military, country flips and timezones)

    
* Step 02: Dropping cities with 0 Fortune 500 company offices. Business goal is to open new offices at locations where Accenture can maximize profit. Our measures of potential profit is the Number of Fortune 500 companies and their combined total profits in the city.

    ```
    # Creating a label/flag column
    df_cities['flag'] = 0
    
    # Adding a new column for number of fortune 500 companies in the location, initializing their count to 0
    df_cities['no_of_fortune_500'] = 0

   
    # Feature engineering
    # changing the 'flag' to 1 for locations where accenture has offices
    from numpy.lib.shape_base import row_stack
    for city in df_acc_cities.City:
      df_cities.loc[df_cities.city_name == city,'flag'] = 1

    # counting the number of fortune 500 companies in a city then changing the value to the count 
    in our 'no_of_fortune_500' column
    # create a dictionary for the df_fortune.CITY with cities as keys and frequency/count as the value
    fortune500_dict = {}

    for row in df_fortune.CITY:
      if row in fortune500_dict:
        fortune500_dict[row] += 1

      else:
        fortune500_dict[row] = 1

    # print(fortune500_dict)

    # now we put the values of the dictionary keys into our dataframe
    list_cities = list(df_cities.city_name)
    for city in list_cities:
      if city.upper() in fortune500_dict:
        df_cities.at[list_cities.index(city), 'no_of_fortune_500'] = fortune500_dict[city.upper()]


    #removing cities with zero fortune500 companies:

    df_cities.drop(df[(df['no_of_fortune_500'] == 0)].index, inplace=True)

    df_cities.head(5)
       
    ```
    
* Step 03: Creating a new dataframe
The dataframe will have df_cities.city_name, df_cities.Flag, df_cities.population from existing datasets and the two new columns no. of fortune 500 companies and sum of fortune 500 profits. We'll do that by creating a copy of the existing us_cities df, removing other columns we won't be using,
then using a dictiomary to loop over the dataset and determine the count for the new features.


    ```
    #Copying the existing df_cities dataframe
    df = df_cities.copy()
    
    #Dropping some columns
    df.drop(['county_name', 'lat', 'lng', 'ranking', 'zips'], axis = 1, inplace = True)
    
    #calculating the sum of fortune500 profits in a city 
    fortune500_profit = {}
    countt = 0
    # rows_list = []

    for city in df_fortune.CITY:

      if city in fortune500_profit:
          # city row index in the dataframe
          # indexx = df_fortune[city].index
          # print(indexx)
          fortune500_dict[city] += df_fortune.profit[countt]
          # print(df_fortune.profit[countt])

      else:
        fortune500_profit[city] = df_fortune.profit[countt]

      countt += 1


    # now we put the values of the dictionary keys into our dataframe
    list_cities = list(df_cities.city_name)
    for city in list_cities:
      if city.upper() in fortune500_profit:
        df_cities.at[list_cities.index(city), 'sum_fortune_profit'] = fortune500_profit[city.upper()]
    
    ```

New DataFrame showing the first 7 rows after Preprocessing
<img width="700" alt="Screen Shot 2024-01-13 at 12 40 29 PM" src="https://github.com/Anni-Bamwenda/Accenture-Project/assets/67605413/b8291115-53b7-427e-8c53-994117148980">


### Splitting the dataset
We'll split our dataset into training and test sets.
Training set: to train our machine learning model. Results will be used on the test set. The training size was 50%
Test set: Use results from training set to predict new office locations (the label). The test size was 50%
In the future might be better to use 70% for training and 30% for testing.

### Machine Learning
We are using logistic regression so our output will be a binary value of 1: Accenture should open a new office or 0: Accenture should not open a new office. Logistic regression model will learn from our training set, then hypertune/modify the features (columns) to give us better predictions.
Model's outcome is mainly based on these factors:
* Population density
* No. of fortune 500 companies in a city
* Population of the city

<img width="742" alt="Screen Shot 2024-01-13 at 12 55 46 PM" src="https://github.com/Anni-Bamwenda/Accenture-Project/assets/67605413/bfa8ff5f-68b5-42ec-90b1-2a89a680c8df">

Image above from https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-logistic-regression/

Code implemented:

    '''
    #X and Y labels for our model
    y = df['flag']
    X = df.drop(['flag', 'city_name', 'state_name', 'Predicted_Flag'], axis = 1)
    # print(X.head(5))

    # Creating train, test and validation datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.50, random_state = 1234)

    print(X_train.shape)
    print(X_test.shape)
    
    # 1. Create the  scikit-learn LogisticRegression model object below and assign to variable 'model'    
    model = sklearn.linear_model.LogisticRegression()

    # 2. Fit the model to the training data below

    model.fit(X_train, y_train)

    # 3. Make predictions on the test data using the predict_proba() method and assign the result to 
    the variable 'probability_predictions' below:
    probability_predictions = model.predict_proba(X_test) 
    
    # 4. Make predictions on the test data using the predict() method and assign the result to the 
    variable 'class_label_predictions' below:
    class_label_predictions = model.predict(X_test)    
    
    '''

## Model Evaluation
Since we are implementing logistic regression, we will evaluate our model using accuracy score and log loss.
Log-loss is indicative of how close the prediction probability is to the corresponding actual/true value. The more the predicted probability diverges from the actual value, the higher is the log loss value. Our model’s log-loss is 47%.
Accuracy score is the proportion of correct predictions over the total predictions. Our model’s accuracy score is 86%.  

    '''
    # 1. Compute the log loss on 'probability_predictions' and save the result to the variable 'l_loss' below
    l_loss = log_loss(y_test, probability_predictions)    

    # 2. Compute the accuracy score on 'class_label_predictions' and save the result to the variable 'acc_score' below
    acc_score = accuracy_score(y_test, class_label_predictions) 
    '''

## Model Output
We will see the 3 cities that our model predicted. We'll also create a plot showing cities with 5 or more fortune 500 companies for visualizations
    
    '''
    # Predicted values:
    print("Predicted Flag:")
    print(class_label_predictions.dtype)
    
    df['Predicted_Flag'] = pd.DataFrame (class_label_predictions, columns = ['predicted_flag'])

    print(df[df['Predicted_Flag'] == 1])

    # Jacksonville, Florida. Fort Worth, Texas. Wichita, Kansas

    '''
    
    
The three cities that our model predicted are Jacksonville, Florida. Fort Worth, Texas. Wichita, Kansas. 

Below is the plot for insights.

<img width="886" alt="Screen Shot 2024-01-13 at 12 44 44 PM" src="https://github.com/Anni-Bamwenda/Accenture-Project/assets/67605413/d30fbd3b-96b6-4581-8c5b-885267b7431c">


## Future Proposal

To get results with higher quality, the following are suggestions to consider in the future:

* Include companies outside the metropolitan city as part of the main city count.
City counties like Los Angeles have a lot of cities that could be merged together in the same metropolitan region. Long Beach, Santa Monica, Pasadena, Beverly Hills etc. could all be merged into Los Angeles metro.
    
* K-mean clustering.
Using K-Means clustering can help narrow down a location in the city where the office should be using features like latitude and longitude, distance     from airport and hotels, mdein income of the area and whether the location is downtown/uptown.

* Including sum of revenue for each fortune 500 company. I didn't have access to their total revunues, but this could be a great feature to add on model training.
  
 

