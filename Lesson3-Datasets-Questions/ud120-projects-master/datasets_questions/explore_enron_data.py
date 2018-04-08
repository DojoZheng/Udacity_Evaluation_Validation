#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

num_total_datasets = len(enron_data)
print "1. Size of the Enron DataSets: {}".format(num_total_datasets)
print "2. Featues in the Enron DataSets: {}".format(len(enron_data["METTS MARK"]))

# Format the Dictionary
import json;
# mmDict = enron_data.items()[0:2]
# mmDictIndent = json.dumps(mmDict, indent=1)
# print mmDictIndent

# How many POIs are there in the E+F dataset?
names = enron_data.keys()
num_poi = 0
for name in names:
	if enron_data[name]['poi'] == 1:
		num_poi += 1

print '3. Number of POIs:{}'.format(num_poi)


# 18. What is the total value of the stock belonging to James Prentice?
print "4. What is the total value of the stock belonging to James Prentice?"
james_prentice_dict = enron_data['PRENTICE JAMES']
james_prentice = json.dumps(james_prentice_dict, indent=1)
print "\tJames Prentice's total stock value: {}".format(james_prentice_dict['total_stock_value'])

# 19. How many email messages do we have from Wesley Colwell to persons of interest?
wesley_colwell_dict = enron_data['COLWELL WESLEY']
wesley_colwell = json.dumps(wesley_colwell_dict, indent=1)
print "5. Number of email messages from Wesley Colwell to poi: {}".format(wesley_colwell_dict['from_this_person_to_poi'])

# 20. What's the value of stock options exercised by Jeffrey K Skilling?
jeffrey_k_skilling_dict = enron_data['SKILLING JEFFREY K']
jeffrey_k_skilling = json.dumps(jeffrey_k_skilling_dict, indent=1)
# print jeffrey_k_skilling
print "6. Value of stock options exercised by Jeffrey K Skilling: {}".format(jeffrey_k_skilling_dict['exercised_stock_options'])

# 27. How many folks in this dataset have a quantified salary? What about a known email address?
import numpy
num_quantified_salary = num_total_datasets
num_known_email_address = num_total_datasets
for name in names:
	if enron_data[name]['salary'] == 'NaN':
		num_quantified_salary -= 1
	if enron_data[name]['email_address'] == 'NaN':
		num_known_email_address -= 1
print "7. Number of folks having a quantified salary: {}".format(num_quantified_salary)
print "8. Number of folks having a known email address: {}".format(num_known_email_address)


# 28. Conversion from Dictionary to Array
from feature_format import *

features = ["salary", 
	"to_messages", 
	"deferral_payments", 
	"total_payments", 
	"exercised_stock_options", 
	"bonus", 
	"restricted_stock",
	"shared_receipt_with_poi", 
	"restricted_stock_deferred", 
	"total_stock_value", 
	"expenses", 
	"loan_advances", 
	"from_messages", 
	"other", 
	"from_this_person_to_poi", 
	"poi",
	"director_fees",
	"deferred_income",
	"long_term_incentive",
	"email_address",
	"from_poi_to_this_person"]

# print features
feature_list = ["poi", "salary", "bonus"]
data_array = featureFormat(enron_data, feature_list)
label, features = targetFeatureSplit(data_array)
# print data_array
# print label
# print features



# 29. How many people in the E+F dataset have 'NaN' for their total payments? 
# What percentage of people in the dataset as a whole is this?
num_nan_total_payments = 0
for name in names:
    if enron_data[name]['total_payments'] == 'NaN':
        num_nan_total_payments += 1
print "9. Number of people having \'Nan\' for their total payments: {}\t \
	Percentage: {}".format(num_nan_total_payments,
                        100 * float(num_nan_total_payments) / num_total_datasets)



# 30. How many POIs in the E+F dataset have 'NaN' for their total payments? 
# What percentage of POI's as a whole is this?
num_poi_nan_total_payments = 0
for name in names:
	if enron_data[name]['poi'] == True:
		if enron_data[name]['total_payments'] == 'NaN':
			num_poi_nan_total_payments += 1
print "10. Number of POIs having NaN for their total payments: {}\tPercentage: {}".format(num_poi_nan_total_payments, 100 * float(num_poi_nan_total_payments) / num_poi)
