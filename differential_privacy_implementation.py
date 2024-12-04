import pydp as dp
from pydp.algorithms.laplacian import BoundedSum, BoundedMean, Count, Max
import pandas as pd
import statistics 
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

# read csv file and create dataframe
csv1 = 'healthcare_dataset.csv'
health_df = pd.read_csv(csv1,sep=",", engine = "python")
decimals = 2    
health_df['Billing Amount'] = health_df['Billing Amount'].apply(lambda x: round(x, decimals))

# create a dataframe with a person redacted
redact_df = health_df.copy()
redact_1_df = redact_df[0:16]
redact_2_df = redact_df[17:]
redact_1_2_df = [redact_1_df, redact_2_df]
redact_df = pd.concat(redact_1_2_df)

# find and display the sum of the complete data frame billing amount
sum_billing_original = round(sum(health_df['Billing Amount'].to_list()),2)
print("Sum of Complete Billing Amount:",sum_billing_original)

# find and display the sum of of the data frame billing amount with a person removed
sum_billing_redact = round(sum(redact_df['Billing Amount'].to_list()),2)
print("Sum of Redacted Billing Amount:",sum_billing_redact)

# example of membership inference attack
billing_amount = sum_billing_original - sum_billing_redact # find difference in billing amount with person removed
billing_amount = round(billing_amount,2) # round billing amount to 2 decimals
found_person = health_df.loc[health_df['Billing Amount'] == billing_amount] # search dataframe for calculated billing amount to find person removed
found_person_name = found_person.iloc[0]['Name']
print("Name of removed person:",found_person_name)
found_person_condition = (health_df.loc[health_df['Name'] == found_person_name])
print(found_person_condition)

# find the sum of complete billing amount using differential privacy
sum_billing_dp_bs = BoundedSum(epsilon= 8.0, lower_bound =  5, upper_bound = 100000000, dtype ='float') 
sum_billing_dp = sum_billing_dp_bs.quick_result(health_df['Billing Amount'].to_list())
sum_billing_dp = round(sum_billing_dp, 2)
print("Differentially Private Sum of Billing Amount:",sum_billing_dp)

# example of membership inference attack failing due to differential privacy
billing_amount_dp = sum_billing_dp - sum_billing_redact # find difference in differentially private billing amount with person removed
billing_amount_dp = round(billing_amount_dp,2) # round billing amount to 2 decimals
found_person = health_df.loc[health_df['Billing Amount'] == billing_amount_dp] # search dataframe for calculated billing amount to find person removed
print("Number of people found using differentially privacy billing amount:", len(found_person))

# example of calculating mean values with and without differential privacy
mean_age_original = round(stat.mean(health_df['Age'].to_list()), 2)
print("Mean of Ages:",mean_age_original)

mean_age_dp_bm = BoundedMean(epsilon= 0.003, lower_bound =  5, upper_bound = 100, dtype ='float') 
mean_age_dp = mean_age_dp_bm.quick_result(health_df['Age'].to_list())
mean_age_dp = round(mean_age_dp, 2)
print("Differentially Private Mean of Ages:",mean_age_dp)