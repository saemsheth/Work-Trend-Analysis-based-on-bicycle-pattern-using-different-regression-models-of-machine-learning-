#importing the library files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

DF = pd.read_csv("finalData.csv") #reading the csv data file

#setting explanatory variables and class variables
X = DF[['Time','MeanTemp','TotalRain','TotalSnow','TotalPrecip','SnowOnGrnd', 'SpdOfMaxGust' ]] 
Y = DF[['TM', 'TH']]  #values that can be used here --- EBM, WBM, EBH, WBH

#finding relation between each explanatory variable and class variable
s = 50
a = 0.4

#Time vs Bicycle Count(Markham St.) plot
plt.figure()

plt.scatter(X['Time'],Y['TM'], edgecolor='k', c="red", s=s, marker="s", alpha=a, label="Data")
plt.xlim([0,24])
plt.ylim([-10, 700])
plt.xlabel("Time")
plt.ylabel("Bicycle Count(Markham st)")
plt.title("Time vs Bicycle Count(Markham St.)")
plt.legend()
plt.show()

#MeanTemp vs Bicycle Count(Markham St.) plot
plt.figure()

plt.scatter(X['MeanTemp'],Y['TM'], edgecolor='k', c="red", s=s, marker="s", alpha=a, label="Data")
plt.xlim([-25,35])
plt.ylim([-10, 700])
plt.xlabel("MeanTemp (C)")
plt.ylabel("Bicycle Count(Markham st)")
plt.title("MeanTemp vs Bicycle Count(Markham St.)")
plt.legend()
plt.show()

#TotalRain vs Bicycle Count(Markham St.) plot 
plt.figure()

plt.scatter(X['TotalRain'],Y['TM'], edgecolor='k', c="red", s=s, marker="s", alpha=a, label="Data")
plt.xlim([-1,32])
plt.ylim([-10, 700])
plt.xlabel("TotalRain (mm)")
plt.ylabel("Bicycle Count(Markham st)")
plt.title("TotalRain vs Bicycle Count(Markham St.)")
plt.legend()
plt.show()

#TotalSnow vs Bicycle Count(Markham St.) plot
plt.figure()

plt.scatter(X['TotalSnow'],Y['TM'], edgecolor='k', c="red", s=s, marker="s", alpha=a, label="Data")
plt.xlim([-2,30])
plt.ylim([-10, 700])
plt.xlabel("TotalSnow (cm)")
plt.ylabel("Bicycle Count(Markham st)")
plt.title("TotalSnow vs Bicycle Count(Markham St.)")
plt.legend()
plt.show()

#TotalPrecip vs Bicycle Count(Markham St.) plot
plt.figure()

plt.scatter(X['TotalPrecip'],Y['TM'], edgecolor='k', c="red", s=s, marker="s", alpha=a, label="Data")
plt.xlim([-2,32])
plt.ylim([-10, 700])
plt.xlabel("TotalPrecip (mm)")
plt.ylabel("Bicycle Count(Markham st)")
plt.title("TotalPrecip vs Bicycle Count(Markham St.)")
plt.legend()
plt.show()

#SnowOnGrnd vs Bicycle Count(Markham St.) plot
plt.figure()

plt.scatter(X['SnowOnGrnd'],Y['TM'], edgecolor='k', c="red", s=s, marker="s", alpha=a, label="Data")
plt.xlim([-2,32])
plt.ylim([-10, 700])
plt.xlabel("SnowOnGrnd (cm)")
plt.ylabel("Bicycle Count(Markham st)")
plt.title("SnowOnGrnd vs Bicycle Count(Markham St.)")
plt.legend()
plt.show()

#SpdOfMaxGust vs Bicycle Count(Markham St.) plot
plt.figure()

plt.scatter(X['SpdOfMaxGust'],Y['TM'], edgecolor='k', c="red", s=s, marker="s", alpha=a, label="Data")
plt.xlim([-2,125])
plt.ylim([-10, 700])
plt.xlabel("SpdOfMaxGust (km/hr)")
plt.ylabel("Bicycle Count(Markham st)")
plt.title("SpdOfMaxGust vs Bicycle Count(Markham St.)")
plt.legend()
plt.show()

#split data into training and testing data in 80-20% ratio 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#linear regression
#training the linear regression model on training data set
linear_regr = LinearRegression()
linear_regr.fit(X_train, y_train) 

# Predict on testing data
y_linear_regr = linear_regr.predict(X_test)

#regression coefficients
print('Coefficients: \n', linear_regr.coef_)


# Plot the results
plt.figure()

plt.scatter(np.array(y_test)[:,0], np.array(y_test)[:,1], edgecolor='k', c="navy", s=s, marker="s", alpha=a, label="Data")
plt.scatter(np.array(y_linear_regr)[:,0], np.array(y_linear_regr)[:,1],edgecolor='k', c="red", s=s, marker="^", alpha=a, label="Linear Regression R2 score=%.2f" % linear_regr.score(X_test, y_test))
plt.xlim([-10, 700])
plt.ylim([-10, 700])
plt.xlabel("TM")
plt.ylabel("TH")
plt.title("Comparing the linear regression predicted output with actual output")
plt.legend()
plt.show()

#normal random forest regressor
regr_rf = RandomForestRegressor(n_estimators=7,random_state=3)
regr_rf.fit(X_train, y_train)

#predict the new data
y_rf = regr_rf.predict(X_test)

#plot data
plt.figure()

plt.scatter(np.array(y_test)[:,0], np.array(y_test)[:,1], edgecolor='k', c="navy", s=s, marker="s", alpha=a, label="Data")
plt.scatter(np.array(y_rf)[:,0], np.array(y_rf)[:,1], edgecolor='k', c="red", s=s, marker="^", alpha=a, label="RF R2 score=%.2f" % regr_rf.score(X_test, y_test))
plt.xlim([-10, 700])
plt.ylim([-10, 700])
plt.xlabel("TM")
plt.ylabel("TH")
plt.title("Comparing random forests predicted output with actual output")
plt.legend()
plt.show()

#multi output regressor
regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=7,random_state=3))
regr_multirf.fit(X_train, y_train)

# Predict on new data
y_multirf = regr_multirf.predict(X_test)

# Plot the results
plt.figure()

plt.scatter(np.array(y_test)[:,0], np.array(y_test)[:,1], edgecolor='k', c="navy", s=s, marker="s", alpha=a, label="Data")
plt.scatter(np.array(y_multirf)[:,0], np.array(y_multirf)[:,1],edgecolor='k', c="red", s=s, marker="^", alpha=a, label="Multi RF R2 score=%.2f" % regr_multirf.score(X_test, y_test))
plt.xlim([-10, 700])
plt.ylim([-10, 700])
plt.xlabel("TM")
plt.ylabel("TH")
plt.title("Comparing the multi-output meta estimator with actual output")
plt.legend()
plt.show()

#checking for model accuracies
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score

#K-fold accuracy for linear regression
linear_regr_comp = LinearRegression()
linear_regr_comp.fit(X, Y)

cv_r2_scores_rf = cross_val_score(linear_regr_comp, X, Y, cv=4,scoring='r2')
print(cv_r2_scores_rf)
print("Mean 4-Fold R Squared: {}".format(np.mean(cv_r2_scores_rf)))

#K-fold accuracy for random forest regressor
regr_rf_comp = RandomForestRegressor(n_estimators=7,random_state=3)
regr_rf_comp.fit(X, Y)

cv_r2_scores_rf = cross_val_score(regr_rf_comp, X, Y, cv=4,scoring='r2')
print(cv_r2_scores_rf)
print("Mean 4-Fold R Squared: {}".format(np.mean(cv_r2_scores_rf)))

#K-fold accuracy for multioutput random forest regressor
multi_rf_comp = MultiOutputRegressor(RandomForestRegressor(n_estimators=7,random_state=3))
multi_rf_comp.fit(X, Y)

cv_r2_scores_rf = cross_val_score(multi_rf_comp, X, Y, cv=4,scoring='r2')
print(cv_r2_scores_rf)
print("Mean 4-Fold R Squared: {}".format(np.mean(cv_r2_scores_rf)))

#finding trend of eastbound and westbound traffic on Huron Street

Z = DF[['EBH','WBH','EBM','WBM']]
#plotting Eastbound and Westbound traffic on Huron St.
plt.figure()

plt.scatter(X['Time'], Z['EBH'], edgecolor='k', c="navy", s=s, marker="s", alpha=a, label="EB")
plt.scatter(X['Time'], Z['WBH'],edgecolor='k', c="red", s=s, marker="^", alpha=a, label="WB")
plt.xlim([-1, 24])
plt.ylim([-10, 600])
plt.xlabel("Time")
plt.ylabel("Bicycle Count")
plt.title("Comparing Eastbound and Westbound on Huron Street")
plt.legend()
plt.show()

#finding trend of eastbound and westbound traffic on Markham Street
plt.figure()

plt.scatter(X['Time'], Z['EBM'], edgecolor='k', c="navy", s=s, marker="s", alpha=a, label="EB")
plt.scatter(X['Time'], Z['WBM'],edgecolor='k', c="red", s=s, marker="^", alpha=a, label="WB")

plt.xlim([-1, 24])
plt.ylim([-10, 600])
plt.xlabel("Time")
plt.ylabel("Bicycle Count")
plt.title("Comparing Eastbound and Westbound on Markham Street")
plt.legend()
plt.show()

#predicting bicycle count based on input parameters
regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=7,random_state=3))
regr_multirf.fit(X, Y)

flag = True
while(flag):
   try:
      user_time = int(input("Enter time (0-23):"))
      user_temp = int(input("Enter temp (in Celsius):"))
      user_rain = int(input("Enter rain (mm):"))
      user_snow = int(input("Enter snow (cm):"))
      user_precip = int(input("Enter precipitation (mm):"))
      user_snowgrnd = int(input("Enter Snow on ground (cm):"))
      user_gustspd = int(input("Enter speed of gust (km/hr):"))
       
      result = regr_multirf.predict([[user_time ,user_temp,user_rain,user_snow,user_precip,user_snowgrnd,user_gustspd]])
      result = result.astype(int)
   
      print("Predicted Total Bicycle Count on Markham Street: ",result[0][0])
      print("Predicted Total Bicycle Count on Huron Street: ",result[0][1])

   except ValueError:
      print("Invalid input!")
    
   cn = input("Want to continue?(y/n)")
   if(cn == 'y'):
      flag = True
   else:
    flag = False

print("Thanks for using!")

