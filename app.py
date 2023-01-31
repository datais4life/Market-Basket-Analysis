# Pattern Prediction with Market Basket Analysis

# Import the necessary packages.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from PIL import Image


st.title("Market Basket Analysis")

st.write("In data analysis, finding patterns in data is common theme. Pattern detection can enable the evaluator to see trends in the data and give recommendations for changing or strengthening that pattern. This analysis will use the market basket analysis method to predict patterns in the telecommunication component data gathered.")

st.write("This analysis method searches for product combinations that are frequently purchased together. This method benefits from using larger data sets so that more transactions can be compared and produce more accurate results. Market basket analysis uses association rules to calculate the relationship between products. The product is a rule, which consists of an antecedent and a consequent. These are just lists of items that are purchased by a customer, with the antecedent occurring first and the consequent being a co-occurrence in the transactions. An itemset or set of items in the antecedent and consequent, is produced for every rule.")

st.write("The components of the association rule are calculated to analyze the itemset. Those components are support, confidence, and lift. Support is how often the itemset appears in the data. Support is calculated by dividing the number of transactions with the itemset by the total number of transactions. Confidence calculates if the itemset is popular with individual or combined sales. Confidence is calculated by dividing the combined transactions by the individual transactions. Lift is the ratio between confidence and support and tells the likelihood of both items being bought together. A diagram of how the association rules of market basket analysis is shown to visualize these calculations.")

image1 = Image.open('association_rule.png')
st.image(image1, caption='Association Rule Diagram', use_column_width=True)

st.subheader("Loading the Data")
st.write("This app is set up to automatically run a given data set unless the user drops a CSV in the file uploader. The analysis is for the given data set, however this is an example of how a similar app can be created to upload a data set on a regular basis for analysis.")
data = st.file_uploader("Upload a CSV")

# Load data from the CSV file & show the first few records.
raw = pd.read_csv('/app/teleco_market_basket.csv', skiprows=lambda x: x % 2) # Remove every other blank row.

st.subheader("Research Question and Purpose")

st.write("A telecommunications company is attempting to understand their customer base in a more detailed manner. The company has a data set containing customer transactions for products that they offer to customers and would like to figure out which products the customers’ favor. A market basket analysis on the data set will be conducted to produce this information.")

st.write("The analysis will answer the question of: What group of products are the customers most likely to purchase? The goal of this analysis is to find a group of products that the customers want to buy. This is a great way to evaluate the products the company offers and determine their value for the business.")

st.write("The biggest assumption of market basket analysis is that there will be overlap of items in the transactions. If the data set were to consist of transactions with no overlap of items, this analysis technique would not produce any results. There must be multiple transactions with the same group of products to calculate the association rules.")

st.write("Using the iloc function in Python, an example transaction is displayed. This is transaction number 25 in the data set, and you can see that the customer purchased six items. The remaining item slots are listed as NaN since they are empty.")
raw.iloc[[25]].T

# Data Cleaning

st.write("The data set is manipulated and cleaned prior to performing the market basket analysis. First, the libraries needed are imported.")

st.code('''# Import the necessary packages.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules''')

st.write("The data set is evaluated using the info, format, and shape commands. This data set consists of 7501 rows of data and 20 data attributes.")

# Dataframe Structure
print(raw.info())
print('{} empty rows.'.format(raw.isna().all(axis=1).sum()))
st.write(raw.shape)

st.write("The head command shows the first five rows of the data set.")
st.write(raw.head())

print(raw.dtypes)
print(len(raw))

st.write("To perform the analysis, the data needs to be set up in lists so that each transaction shows all the items purchased on the same row. A simple loop is created to accomplish this.")

# Converting dataframe into list of lists.
outside = len(raw)
inside = raw.shape[1]
transactions = []
for i in range(0,outside):
    transactions.append([str(raw.values[i,j]) 
        for j in range(0,inside)])
TE = TransactionEncoder()
array = TE.fit_transform(transactions)

st.code('''# Converting dataframe into list of lists.
outside = len(raw)
inside = raw.shape[1]
transactions = []
for i in range(0,outside):
    transactions.append([str(raw.values[i,j]) 
        for j in range(0,inside)])
TE = TransactionEncoder()
array = TE.fit_transform(transactions)''')

# Show the list to verify the change.
print(TE.columns_)
print(len(TE.columns_))

st.write("There are now 120 columns or transactions in the data frame.")
array.shape

st.write("Now, the null values can be dropped using the drop command to create the final clean data set of 7501 rows and 119 columns.")

# Create cleaned dataframe.
df = pd.DataFrame(array, columns=TE.columns_)
df = df.drop(['nan'],axis=1) #drop 'nan' blank column
print(df.info())
print(df.shape)

st.code('''# Create cleaned dataframe.
df = pd.DataFrame(array, columns=TE.columns_)
df = df.drop(['nan'],axis=1) #drop 'nan' blank column''')

#df.head()

st.write("The info and head commands show the cleaned data frame’s first four columns.")

# Final dataframe ready for market basket analysis.
#df.info()
df.head(4).T

st.subheader("Analysis")
st.write("To perform the analysis, first the outputs of antecedents, consequents, support, confidence, and lift are defined.")

# Data Exploration

# Association Rules
# Define which columns to include in printouts.
output_cols = ['antecedents','consequents','support','confidence','lift']

st.code('''# Association Rules
# Define which columns to include in printouts.
output_cols = ['antecedents','consequents','support','confidence','lift']''')

st.write("Using MLXtends frequent_patterns package, the itemsets and rules are identified using the minimum support of 4/1000 and sorted by lift.")

st.code('''# min_sup 4/1000 - min_lift 1 (sorted by lift)
itemsets = apriori(df, min_support=4/1000, use_colnames=True)
rules = association_rules(itemsets, metric='lift', min_threshold=1)
rules[output_cols].sort_values(by=['lift'], ascending=False).round(3).head(20)''')

# min_sup 4/1000 - min_lift 1 (sorted by lift)
itemsets = apriori(df, min_support=4/1000, use_colnames=True)
rules = association_rules(itemsets, metric='lift', min_threshold=1)
st.write(rules[output_cols].sort_values(by=['lift'], ascending=False).round(3).head(20))

st.write("This process is re-run using the parameters of 3/1000 for the minimum support and the minimum confidence of 3/10.")

st.code('''# min_sup 3/1000 - min_conf 3/10 (sorted by lift)
pd.set_option('display.max_colwidth', None)
itemsets = apriori(df, min_support=3/1000, use_colnames=True)
rules = association_rules(itemsets, metric='confidence', min_threshold=3/10)
rules[output_cols].sort_values(by=['lift'], ascending=False).round(3).head(8)''')

# min_sup 3/1000 - min_conf 3/10 (sorted by lift)
pd.set_option('display.max_colwidth', None)
itemsets = apriori(df, min_support=3/1000, use_colnames=True)
rules = association_rules(itemsets, metric='confidence', min_threshold=3/10)
st.write(rules[output_cols].sort_values(by=['lift'], ascending=False).round(3).head(8))

st.write("In reviewing the results, the items of “Dust-off Compressed Gas 2 Pack” and “VIVO Dual LCD Monitor Desk Mount” are identified the most, so these items are defined. Running the itemset on these two items shows that they are bought together 448 times in this data frame.")

st.code('''item1 = 'Dust-Off Compressed Gas 2 pack'
item2 = 'VIVO Dual LCD Monitor Desk mount'
itemset = [item1, item2]
print('item1: ', len(df[itemset][(df[item1] == True) ]))
print('item2: ', len(df[itemset][(df[item2] == True) ]))
print('both: ', len(df[itemset][(df[item1] == True) & (df[item2] == True)]))''')

item1 = 'Dust-Off Compressed Gas 2 pack'
item2 = 'VIVO Dual LCD Monitor Desk mount'
itemset = [item1, item2]
st.write('item1: ', len(df[itemset][(df[item1] == True) ]))
st.write('item2: ', len(df[itemset][(df[item2] == True) ]))
st.write('both: ', len(df[itemset][(df[item1] == True) & (df[item2] == True)]))

st.write("Running the Apriori algorithm again with the minimum support and confidence of 1/100 and sorted by support is run. It shows that there is a support value of 0.06 in this data set, which is a very low score.")

st.code('''# min_sup 1/100 - min_conf 1/100 (sorted by support)
pd.set_option('display.max_colwidth', None)
itemsets = apriori(df, min_support=1/100, use_colnames=True)
rules = association_rules(itemsets, metric='confidence', min_threshold=1/100)
rules[output_cols].sort_values(by=['support'], ascending=False).round(5).head(10)''')

# min_sup 1/100 - min_conf 1/100 (sorted by support)
pd.set_option('display.max_colwidth', None)
itemsets = apriori(df, min_support=1/100, use_colnames=True)
rules = association_rules(itemsets, metric='confidence', min_threshold=1/100)
st.write(rules[output_cols].sort_values(by=['support'], ascending=False).round(5).head(10))

st.write("These same parameters are run again but are sorted by confidence score. It shows a confidence score of 0.51, which is a mid-range value since the range is between 0 and 1.")

st.code('''# min_sup 1/100 - min_conf 1/100 (sorted by confidence)
pd.set_option('display.max_colwidth', None)
itemsets = apriori(df, min_support=1/100, use_colnames=True)
rules = association_rules(itemsets, metric='confidence', min_threshold=1/100)
rules[output_cols].sort_values(by=['confidence'], ascending=False).round(3).head(20)''')

# min_sup 1/100 - min_conf 1/100 (sorted by confidence)
pd.set_option('display.max_colwidth', None)
itemsets = apriori(df, min_support=1/100, use_colnames=True)
rules = association_rules(itemsets, metric='confidence', min_threshold=1/100)
st.write(rules[output_cols].sort_values(by=['confidence'], ascending=False).round(3).head(20))

st.write("The algorithm is run again and sorted by lift score. This shows a stronger lift value of 3.29 with the combination of the 128GB SanDisk card and the 64GB SanDisk card.")

st.code('''# min_sup 1/100 - min_conf 1/100 (sorted by lift)
pd.set_option('display.max_colwidth', None)
itemsets = apriori(df, min_support=1/100, use_colnames=True)
rules = association_rules(itemsets, metric='confidence', min_threshold=1/100)
rules[output_cols].sort_values(by=['lift'], ascending=False).round(3).head()''')

# min_sup 1/100 - min_conf 1/100 (sorted by lift)
pd.set_option('display.max_colwidth', None)
itemsets = apriori(df, min_support=1/100, use_colnames=True)
rules = association_rules(itemsets, metric='confidence', min_threshold=1/100)
st.write(rules[output_cols].sort_values(by=['lift'], ascending=False).round(3).head())

st.write("To identify the top three rules for the analysis, first the data is pruned to count the number of times the itemset occurs in the data frame to weed out infrequent itemsets. There are now 432 rules to analyze.")

st.code('''# pruning
pruned_rules = rules[rules['confidence']>0.02]
print('only {} pru_r are left.'.format(len(pruned_rules)))''')

# pruning
pruned_rules = rules[rules['confidence']>0.02]
st.write('only {} pru_r are left.'.format(len(pruned_rules)))

st.write("Running the analysis and sorting for lift, the top three rules are summarized. The best rule produced has a lift value of 6.116, which is a strong indication that the itemset would be purchased together. This itemset includes the Dust-Off Compressed Gas 2 pack, Ankler 2-in-1 USB Card Reader, and FEIYOLD Blue light Blocking Glasses.")

st.code('''# top 3 rules sorted by lift
pd.set_option('display.max_colwidth', None)
itemsets = apriori(df, min_support=3/1000, use_colnames=True)
rules = association_rules(itemsets, metric='confidence', min_threshold=4/10)
rules[output_cols].sort_values(by=['lift'], ascending=False).round(3).head(3)''')

# top 3 rules sorted by lift
pd.set_option('display.max_colwidth', None)
itemsets = apriori(df, min_support=3/1000, use_colnames=True)
rules = association_rules(itemsets, metric='confidence', min_threshold=4/10)
st.write(rules[output_cols].sort_values(by=['lift'], ascending=False).round(3).head(3))

itemsets = apriori(df, min_support=3/1000, use_colnames=True)
rules = association_rules(itemsets, metric='lift', min_threshold=5)

rules['lhs items'] = rules['antecedents'].apply(lambda x:len(x) )
rules[rules['lhs items']>1].sort_values('lift', ascending=False).head()

# Replace frozen sets with strings
rules['antecedents_'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['consequents_'] = rules['consequents'].apply(lambda a: ','.join(list(a)))

st.write("The data frame of rules is transformed into a matrix using the lift metric. This is in preparation of generating a heatmap to analyze the data.")

# Transform the DataFrame of rules into a matrix using the lift metric
pivot = rules[rules['lhs items']>1].pivot(index = 'antecedents_', columns = 'consequents_', values= 'lift')
rules

st.code('''# Transform the DataFrame of rules into a matrix using the lift metric
pivot = rules[rules['lhs items']>=1].pivot(index = 'antecedents_', columns = 'consequents_', values= 'lift')
pivot''')

# Transform the DataFrame of rules into a matrix using the lift metric
pivot = rules[rules['lhs items']>=1].pivot(index = 'antecedents_', columns = 'consequents_', values= 'lift')
st.write(pivot)

st.write("The final product of the analysis produces a heatmap which is shown and saved as market_basket.png on the user's machine.")

# Generate a heatmap with annotations on and the colorbar off
sns.heatmap(pivot, annot = True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.savefig("market_basket.png")
#plt.show()
image2 = Image.open('market_basket.png')
st.image(image2, caption='Heatmap of Market Basket Analysis', use_column_width=True)

st.subheader("Data Summary and Implications")
st.write("Association Rule Significance: The top rule that the analysis produced is shown in the figure. From this, the support for this rule is a value of 0.004 can be derived. This is calculated by dividing the number of transactions with Dust-Off Compressed Gas 2 pack, Ankler 2-in-1 USB Card Reader, and FEIYOLD Blue light Blocking Glasses by the total number of transactions.")
st.write("The confidence value of this itemset is calculated at 0.403, which equates to 40.3 percent of the transactions that had Dust-Off Compressed Gas 2 pack and Ankler 2-in-1 USB Card Reader as the antecedent also had the FEIYOLD Blue light Blocking Glasses as the consequent.")
st.write("The lift value of this rule is strong with a score of 6.116. This states that a customer is six times more likely to purchase the FEIYOLD Blue light Blocking Glasses if they are also buying the Dust-Off Compressed Gas 2 pack and “Ankler 2-in-1 USB Card Reader” together.")
st.write("Practical Significance: Market basket analysis is typically performed on very large data sets; however, this data set was only 7501 transactions. This made the calculations limited and caused a low support value to be calculated for even the top rules. Since the confidence level was calculated at 40.3 percent for the top rule, the rule should still be considered a good itemset combination and strong association.")
st.write("Recommendations: The product placement for FEIYOLD Blue light Blocking Glasses should be next to the computer accessories. Specifically, the blue light glasses should be placed next to the Dust-Off Compressed Gas 2 pack and Ankler 2-in-1 USB Card Reader. That same display should be next to the selection of memory cards. This will increase the likelihood of these items being purchased together and increasing sales. This will also improve customer service for the ease of finding what they need in the store.")