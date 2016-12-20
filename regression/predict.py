from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron
import pandas
import numpy
import matplotlib as plt
from sklearn.pipeline import make_pipeline
from sklearn import linear_model, feature_extraction
def categorical_features(row):
    d = {}
    d["STATE"] = row[1]["STATE"]
    return d

def last_poll(full_data):
    """
    Create feature from last poll in each state
    """
    
    # Only care about republicans
    repub = full_data[full_data["PARTY"] == "Rep"]
    
    # Sort by date
    chron = repub.sort_values(by="DATE", ascending=True)
    
    # Only keep the last one
    dedupe = chron.drop_duplicates(subset="STATE", keep="last")
    
    # Remove national polls
    return dedupe[dedupe["STATE"] != "US"]

if __name__ == "__main__":
    # Read in the X data
    all_data = pandas.read_csv("data.csv")
    
    # Remove non-states
    all_data = all_data[pandas.notnull(all_data["STATE"])]
    
    # split between testing and training
    train_x = last_poll(all_data[all_data["TOPIC"] == '2012-president'])
    train_x.set_index("STATE")
    
    test_x = last_poll(all_data[all_data["TOPIC"] == '2016-president'])
    test_x.set_index("STATE")
    
    # Read in the Y data
    y_data = pandas.read_csv("../data/2012_pres.csv", sep=';')
    y_data = y_data[y_data["PARTY"] == "R"]
    y_data = y_data[pandas.notnull(y_data["GENERAL %"])]
    y_data["GENERAL %"] = [float(x.replace(",", ".").replace("%", ""))
    
    for x in y_data["GENERAL %"]]
    y_data["STATE"] = y_data["STATE ABBREVIATION"]
    y_data.set_index("STATE")
   
    backup = train_x
    train_x = y_data.merge(train_x, on="STATE",how='left')
   
    # make sure we have all states in the test data
    for ii in set(y_data.STATE) - set(test_x.STATE):
        new_row = pandas.DataFrame([{"STATE": ii}])
        test_x = test_x.append(new_row)

    # format the data for regression
    train_x = pandas.concat([train_x.STATE.astype(str).str.get_dummies(),train_x], axis=1)
    test_x = pandas.concat([test_x.STATE.astype(str).str.get_dummies(),test_x], axis=1)
        
    # handle missing data
    for dd in train_x, test_x:
        dd["NOPOLL"] = pandas.isnull(dd["VALUE"])
        dd["VALUE"] = dd["VALUE"].fillna(0.0)
        dd["NOMOE"] = pandas.isnull(dd["MOE"])
        dd["MOE"] = dd["MOE"].fillna(0.0)

    # create feature list
    features = list(y_data.STATE)
    features.append("VALUE")
    features.append("NOPOLL")
    features.append("MOE")
    features.append("NOMOE")
    
    features_rep = list(y_data.PARTY)# from y data
    features_rep = [ord(i) for i in features_rep]# changing R into ascii value
    features_obs = list(all_data.OBS)#from  csv all_data since its not in y data
    features_val = list(all_data.VALUE)# adding value as features
    features_moe = list(all_data.MOE)
    
    feature_matrix=[] #creating empty matrix

    for i in range(len(features_rep)):
        feature_matrix.append([features_rep[i],features_obs[i],features_val[i],features_moe[i]])
        # appending to list R and OBS
    features_matrix=numpy.array(feature_matrix)#returning an array with value of R and OBs
    
    
    feature_matrix1=[] #creating empty matrix
    for i in range(len(features_rep)):
        feature_matrix1.append([features_rep[i],features_obs[i],features_val[i],features_moe[i]])# appending to list R and OBS
    features_matrix1=numpy.array(feature_matrix1)#returning an array with value of R and OBs


    # fit the regression
    #mod = linear_model.LinearRegression()
    #mod = linear_model.Ridge(alpha=.667)
    #mod = linear_model.BayesianRidge()
    #mod.fit(train_x[features], train_x["GENERAL %"])


    #pipelining
    mod=make_pipeline(PolynomialFeatures(degree=6),linear_model.LinearRegression())# making a model for polynomial regression
    mod.fit(feature_matrix, train_x["GENERAL %"])# passing matris with feature
    
    #modd=make_pipeline(PolynomialFeatures(degree=4),linear_model.BayesianRidge())# making a model for polynomial regression
    #modd.fit(feature_matrix1, train_x["GENERAL %"])# passing matris with feature
    
    modd=make_pipeline(PolynomialFeatures(degree=6),linear_model.LinearRegression())# making a model for polynomial regression
    modd.fit(feature_matrix1, train_x["GENERAL %"])# passing matris with feature
    
    
    # Write out the model
    #with open("model.txt", 'w') as out:
    #out.write("BIAS\t%f\n" % mod.intercept_)
    #for jj, kk in zip(features, mod.coef_):
    #out.write("%s\t%f\n" % (jj, kk))
    
    # Write the predictions
    pred_test = mod.predict(features_matrix)
    with open("pred.txt", 'w') as out:
        for ss, vv in sorted(zip(list(test_x.STATE), pred_test)):
            out.write("%s\t%f\n" % (ss, vv))

# Write the predictions
    pred_test = modd.predict(feature_matrix1)
    with open("pred.txt", 'w') as out:
        for ss, vv in sorted(zip(list(test_x.STATE), pred_test)):
            out.write("%s\t%f\n" % (ss, vv))

print("Mean squared error: %.2f"
      % numpy.mean((mod.predict(features_matrix) - train_x["GENERAL %"]) ** 2))
#test data - train data
#(prediction - observed)^2
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % mod.score(features_matrix,train_x["GENERAL %"]))

print("Mean squared error2: %.2f"
      % numpy.mean((modd.predict(features_matrix1) - train_x["GENERAL %"]) ** 2))
#test data - train data
#(prediction - observed)^2

# Explained variance score: 1 is perfect prediction
print('Variance score2: %.2f' % modd.score(features_matrix1,train_x["GENERAL %"]))
