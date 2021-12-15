#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns; sns.set()
import datetime 
from datetime import date
import calendar

#reading file
pos = pd.read_csv('supermarket_sales.csv')

#initial index to 1
pos.index += 1

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

def img_to_pie( fn, wedge, xy, zoom=1, ax = None):
    if ax==None: ax=plt.gca()
    im = plt.imread(fn, format='png')
    path = wedge.get_path()
    patch = PathPatch(path, facecolor='#90ee90')
    ax.add_patch(patch)
    imagebox = OffsetImage(im, zoom=zoom, clip_path=patch, zorder=-10)
    ab = AnnotationBbox(imagebox, xy, xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:%Y-%m-%d}, y={:.2f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.5", fc="w", ec="k", lw=1)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=15", color='navy')
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.8,0.9), **kw, color='navy')
    
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

#Change to datetime format for Date and Time column
new_pos['Date'] = pd.to_datetime(new_pos['Date'], errors='coerce')
new_pos['Time'] = pd.to_datetime(new_pos['Time'])

#month sales
total_month_sales = new_pos.loc[:, ['Branch', 'Product line', 'Total','Date']]
total_month_sales['Month'] = total_month_sales['Date'].dt.month
results = total_month_sales.groupby('Month').sum()
months = range(1,4)
color = ['red', 'pink', 'purple']
plt.bar(months,results['Total'], width = 0.5, color = color)
my_xticks = ['Jan','Feb','Mar']
plt.xticks(months, my_xticks)
plt.ylabel('Total sales ($)')
plt.xlabel('Month')
plt.title("Month sales", fontsize=16, color = 'k')
plt.show()

#Branch sales for each month
results = total_month_sales.groupby(['Month','Branch']).sum()
results.unstack().plot()
plt.ylabel(' Total sales ($)')
plt.title("Branch sales for each month", fontsize=16, color = 'k')
legend = ['Branch A', 'Branch B', 'Branch C']
plt.legend(legend, bbox_to_anchor=(1,1), loc="upper left")
plt.show()

#Demand for product line
total_month_sales = new_pos.loc[:, ['Branch', 'Product line', 'Quantity']]
results = total_month_sales.groupby(['Product line','Branch']).sum()
quantity_branch = results.unstack()
legend_color = ['red', 'pink', 'purple']
quantity_branch.plot(kind='barh', color = legend_color)
plt.xlabel('Quantity')
legend = ['Branch A', 'Branch B', 'Branch C']
plt.legend(legend, bbox_to_anchor=(1,1), loc="upper left")
plt.title('Demand for product line', fontsize=16, color = 'k')
plt.show()

#Product sales
product_line = new_pos.groupby('Product line')
quantity = product_line.sum()['Quantity']
products = [product for product, df in product_line]
color = ['red', 'pink', 'purple', 'orange', 'green', 'blue']
plt.barh(products, quantity, color = color)
plt.xlabel('Sales quantity')
plt.title("Product sales", fontsize=16, color = 'k')
plt.show()

#sales by gender
male = new_pos['Gender'][new_pos['Gender'].str.contains('Male')].count()
female = new_pos['Gender'][new_pos['Gender'].str.contains('Female')].count()
gender = np.array([male, female])
labels = ['Male', 'Female']
plt.gca().axis("equal")
wedges, texts = plt.pie(gender, labels = labels, startangle = 90, wedgeprops = { 'linewidth': 2, "edgecolor" :"k","fill":False})
plt.title("Gender", fontsize=16, color = 'k')

positions = [(-0.4,0),(0.43,0)]
zooms = [0.27,0.053]

for i in range(2):
    fn = "{}.png".format(labels[i].lower())
    img_to_pie(fn, wedges[i], xy=positions[i], zoom=zooms[i] )
    wedges[i].set_zorder(10)
    
plt.show()

#rating for each branch #box plot
colors = ('Purple', 'Blue', 'Red')
sns.boxplot(data=new_pos, x='Branch', y='Rating',order=["A", "B","C"], width = 0.5, palette=colors, medianprops=dict(color="gold")).set(title = 'Branch rating')
plt.show()

#Product purchases
sns.boxplot(data=new_pos, x='Quantity', y='Product line', palette = 'Spectral', width = 0.5).set(title = 'Product purchases')
plt.show()

#Member demographic
member = new_pos['Customer type'][new_pos['Customer type'].str.contains('Member')].count()
non_member = new_pos['Customer type'][new_pos['Customer type'].str.contains('Normal')].count()
customer_type = np.array([member, non_member])
member_labels = ['Member', 'Non member']
plt.gca().axis("equal")
wedges, texts = plt.pie(customer_type, labels = member_labels, startangle = 90, wedgeprops = { 'linewidth': 2, "edgecolor" :"k","fill":False})
plt.title('Membership', fontsize=16, color = 'k')

positions = [(-0.5,0),(0.5,0)]
zooms = [0.12,0.35]

for i in range(2):
    fn = "{}.png".format(member_labels[i].lower())
    img_to_pie(fn, wedges[i], xy=positions[i], zoom=zooms[i] )
    wedges[i].set_zorder(10)
    
plt.show()

#Payment type
cash = new_pos['Payment'][new_pos['Payment'].str.contains('Cash')].count()
ewallet = new_pos['Payment'][new_pos['Payment'].str.contains('Ewallet')].count()
credit_card = new_pos['Payment'][new_pos['Payment'].str.contains('Credit card')].count()
payment_type = np.array([cash, ewallet, credit_card])
payment_labels = ['Cash', 'E-wallet', ' Credit card']
colors = ('cyan', 'Aquamarine', 'gold')
explodeTuple = (0.1, 0.0, 0.0)
fig, ax = plt.subplots(figsize=(6, 6))
patches, texts, pcts = ax.pie(payment_type, labels = payment_labels, colors = colors, explode=explodeTuple, pctdistance=0.75, autopct='%1.2f%%', shadow = True, startangle=120)
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
for i, patch in enumerate(patches):
  texts[i].set_color(patch.get_facecolor())
plt.setp(pcts, color='Navy')
plt.setp(texts, fontweight=300)
ax.set_title('Payment Type', fontsize=16, color = 'Red')
plt.tight_layout()
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()

#Product Sales per Hour
new_pos['Hour'] = new_pos['Time'].dt.hour
new_pos['Hour'].unique()
find_total_quantity = new_pos.groupby(['Hour']).sum()
sns.lineplot(x = 'Hour',  y = 'Quantity',data = find_total_quantity, color = '#D100D1').set_title("Product Sales per Hour")
plt.show()

#Product Sales per Hour for each branch
find_total_quantity = new_pos.groupby(['Hour','Branch']).sum()
sns.lineplot(x = 'Hour',  y = 'Quantity',data = find_total_quantity, hue = 'Branch', palette="flare").set_title("Branch Product Sales per Hour")
plt.show()

#Demand for product line
total_month_sales = new_pos.loc[:, ['Branch', 'Product line', 'Quantity']]
results = total_month_sales.groupby(['Product line','Branch']).sum()
quantity_branch = results.unstack()
legend_color = ['red', 'pink', 'purple']
quantity_branch.plot(kind='barh', color = legend_color)
plt.xlabel('Quantity')
legend = ['Branch A', 'Branch B', 'Branch C']
plt.legend(legend, bbox_to_anchor=(1,1), loc="upper left")
plt.title('Demand for product line', fontsize=16, color = 'k')
plt.show()

#Gross income by product
product_income=new_pos[["Product line", "gross income"]].groupby(['Product line'], as_index=False).sum().sort_values(by='gross income', ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x='Product line', y='gross income', data=product_income, palette = "hls")
plt.title('Gross income by product', fontsize=16, color = 'k')
plt.show()

#Product cost
product_cogs=new_pos[["Product line", "cogs"]].groupby(['Product line'], as_index=False).sum().sort_values(by='cogs', ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x='Product line', y='cogs', data=product_cogs, palette = "Accent")
plt.title('Product cost', fontsize=16, color = 'k')
plt.show()

#Correlation between variables
sns.heatmap(np.round(new_pos.corr(),2), annot=True, cmap = 'Wistia')
plt.title('Correlation between variables', fontsize=14, color = 'k')
plt.show()

#Customer Rating Distribution
sns.displot(new_pos['Rating'], kde=True, color = 'pink')
plt.axvline(x=np.mean(new_pos['Rating']), c='red', ls='--', label='mean')
plt.axvline(x=np.percentile(new_pos['Rating'],25),c='green', ls='--', label = '25th percentile:Q1')
plt.axvline(x=np.percentile(new_pos['Rating'],75),c='orange', ls='--',label = '75th percentile:Q3' )
plt.title("Customer Rating Distribution", fontsize=16, color = 'k')
plt.legend(bbox_to_anchor=(1,1))
plt.show()

#Gross Income for Each Branch
sns.boxplot(x=new_pos['Branch'], order = ['A','B','C'], y=new_pos['gross income'], palette = 'autumn')
plt.ylabel('Gross income ($)')
plt.xlabel('Branch')
plt.title("Gross Income for Each Branch", fontsize=16, color = 'k')
plt.show()

#Gross Income for Each Branch (Line Graph)
branch_monthly_grossincome = new_pos.loc[:, ['Branch', 'gross income','Date']]
branch_monthly_grossincome['Month'] = branch_monthly_grossincome['Date'].dt.month_name().str.slice(stop=3)
months = ["Jan", "Feb", "Mar"]
branch_monthly_grossincome['Month'] = pd.Categorical(branch_monthly_grossincome['Month'], categories=months, ordered=True)
branch_monthly_grossincome.sort_values('Month')
results = branch_monthly_grossincome.groupby(['Month','Branch']).sum()

results.unstack().plot(color=['red','blue','magenta'], marker='o')
plt.ylabel('Gross income ($)')

legend = ['Branch A', 'Branch B', 'Branch C']
plt.legend(legend, bbox_to_anchor=(1,1), loc="upper left")
plt.title("Gross Income for Each Branch Across Time", fontsize=16, color = 'k')
plt.show()

#Gender and Gross Income Distribution
sns.boxplot(x=new_pos['Gender'], y=new_pos['gross income'], palette = 'prism')
plt.ylabel('Gross income ($)')
plt.title("Gross Income Distribution based on Gender", fontsize=16, color = 'k')
plt.show()

#Time trend in gross income
fig, ax = plt.subplots(figsize=(9,9))
date = new_pos.groupby(new_pos.Date).mean().index
gross_income = new_pos.groupby(new_pos.Date).mean()['gross income']
sns.lineplot(x= date, y = gross_income, ax=ax, color='red')

annot_max(date, gross_income)
plt.title('Gross income', fontsize=16, color = 'k')
plt.tight_layout
plt.show()

#Spending pattern based on gender
plt.figure(figsize=(12, 6))
plt.title('Total Monthly transaction by Gender')
sns.countplot(x=new_pos['Product line'], hue = new_pos.Gender, palette='coolwarm')
plt.show()

#Which day of the week has maximum sales 
new_pos['Weekday'] = new_pos['Date'].dt.day_name()
weekday_sorted = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
new_pos['Weekday'] = pd.Categorical(new_pos['Weekday'], categories=weekday_sorted, ordered=True)
new_pos = new_pos.sort_values('Weekday')

plt.figure(figsize=(8, 6))
plt.title('Daily Sales by Day of the Week')
sns.countplot(x=new_pos['Weekday'], palette = 'Spectral')
plt.show()

#Rating of products
xdata = [0,1,2,3,4,5,6,7,8,9,10]
plt.figure(figsize = (9,6))
sns.barplot(y = new_pos['Product line'], x = new_pos['Rating'], palette = "Paired")
plt.xticks(xdata)
plt.show()

#Quantity purchased by product
sns.boxenplot(y = 'Product line', x = 'Quantity', data=new_pos, palette = "rainbow" )
plt.show()
