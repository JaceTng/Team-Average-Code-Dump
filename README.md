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
color = ['red', 'pink', 'purple']
plt.bar(months,results['Total'], width = 0.5, color = color)
plt.xticks(months)
plt.ylabel('Total sales ($)')
plt.xlabel('Month')
plt.title("Month sales", fontsize=16, color = 'k')
plt.show()

#Branch sales for each month
results = total_month_sales.groupby(['Month','Branch']).sum()
results.unstack().plot()
plt.ylabel(' Total sales ($)')
plt.title("Branch sales for each month", fontsize=16, color = 'k')
plt.show()

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

def img_to_pie( fn, wedge, xy, zoom=1, ax = None):
    if ax==None: ax=plt.gca()
    im = plt.imread(fn, format='png')
    path = wedge.get_path()
    patch = PathPatch(path, facecolor='none')
    ax.add_patch(patch)
    imagebox = OffsetImage(im, zoom=zoom, clip_path=patch, zorder=-10)
    ab = AnnotationBbox(imagebox, xy, xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

positions = [(-0.5,0),(0.5,0)]
zooms = [0.2,0.1]

for i in range(2):
    fn = "{}.png".format(labels[i].lower())
    img_to_pie(fn, wedges[i], xy=positions[i], zoom=zooms[i] )
    wedges[i].set_zorder(5)
    
plt.show()

#rating for each branch #box plot
colors = ('Purple', 'Blue', 'Red')
box = sns.boxplot(data=new_pos, x='Branch', y='Rating', width = 0.5, palette=colors, medianprops=dict(color="gold")).set(title = 'Branch rating')
plt.show()

#Product purchases
sns.boxplot(data=new_pos, x='Quantity', y='Product line', width = 0.5).set(title = 'Product purchases')
plt.show()

member = new_pos['Customer type'][new_pos['Customer type'].str.contains('Member')].count()
non_member = new_pos['Customer type'][new_pos['Customer type'].str.contains('Normal')].count()
customer_type = np.array([member, non_member])
member_labels = ['Member', 'Non member']
plt.pie(customer_type, labels = member_labels, startangle = 90)
plt.title('Membership', fontsize=16, color = 'k')
plt.show()

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
new_pos['Time'] = pd.to_datetime(new_pos['Time'])
new_pos['Hour'] = (new_pos['Time']).dt.hour
new_pos['Hour'].unique()
find_total_quantity = new_pos.groupby(['Hour']).sum()
sns.lineplot(x = 'Hour',  y = 'Quantity',data = find_total_quantity).set_title("Product Sales per Hour")
plt.show()
#Product Sales per Hour for each branch
find_total_quantity = new_pos.groupby(['Hour','Branch']).sum()
sns.lineplot(x = 'Hour',  y = 'Quantity',data = find_total_quantity, hue = 'Branch', palette="flare").set_title("Branch Product Sales per Hour")
plt.show()

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
