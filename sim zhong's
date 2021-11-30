#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading file
pos = pd.read_csv('supermarket_sales.csv')

#initial index to 1
pos.index += 1
pos.head()

# to find missing values
def find_quantity():
    quantity = (new_pos['gross income'] + new_pos['cogs'])/new_pos['Unit price']
    return quantity
def find_unit_price():
    unit_price = (new_pos['gross income'] + new_pos['cogs'])/new_pos['Quantity']
    return unit_price
def find_gross_income():
    gross_income = new_pos['Unit price'] * new_pos['Quantity'] - new_pos['cogs']
    return gross_income
def find_cogs():
    cogs = new_pos['Unit price'] * new_pos['Quantity'] - new_pos['gross income']
    return cogs
    
#check for specific data
a = ['565-91-4567']
new_pos[new_pos['Invoice ID'].isin(a)]
    
#dropping entire rows with NaN/NA for branch, product line and date
new_pos = pos.dropna(how='any', subset=['Branch','Product line', 'Date'])

#populating NA value in rating with mean
new_pos['Rating'].fillna(new_pos['Rating'].mean(), inplace=True)
new_pos['Rating'] = new_pos['Rating'].round(1)

#front fill - NaN data replaced by 1 male 1 female
new_pos['Gender'].fillna(method='ffill', inplace=True)

#filling data
new_pos['Quantity'].fillna(find_quantity(), inplace=True)
new_pos['Unit price'].fillna(find_unit_price(), inplace=True)
new_pos['gross income'].fillna(find_gross_income(), inplace=True)
new_pos['cogs'].fillna(find_cogs(), inplace=True)
