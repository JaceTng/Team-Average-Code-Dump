#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

#month sales
total_month_sales = new_pos.loc[:, ['Branch', 'Product line', 'Total','Date']]
total_month_sales['Month'] = total_month_sales['Date'].str[0:1]
results = total_month_sales.groupby('Month').sum()
months = range(1,4)
plt.bar(months,results['Total'], width = 0.5)
plt.xticks(months)
plt.ylabel('Total sales ($)')
plt.xlabel('Month')
plt.show()

#Branch sales for each month
results = total_month_sales.groupby(['Month','Branch']).sum()
results.unstack().plot()
plt.ylabel(' Total sales ($)')
plt.show()

product_line = new_pos.groupby('Product line')
quantity = product_line.sum()['Quantity']
products = [product for product, df in product_line]
plt.bar(products, quantity)
plt.xticks(products, rotation = 'vertical')
plt.ylabel(' Total Sales quantity')
plt.show()

#sales by gender
male = new_pos['Gender'][new_pos['Gender'].str.contains('Male')].count()
female = new_pos['Gender'][new_pos['Gender'].str.contains('Female')].count()
gender = np.array([male, female])
gender_labels = ['Male', 'Female']
plt.pie(gender, labels = gender_labels, startangle = 90)
plt.show()

#rating for each branch #box plot
sns.boxplot(data=new_pos, x='Branch', y='Rating', width = 0.5)
plt.show()

#unit price for each product
sns.boxplot(data=new_pos, x='Unit price', y='Product line', width = 0.5)
plt.show()
