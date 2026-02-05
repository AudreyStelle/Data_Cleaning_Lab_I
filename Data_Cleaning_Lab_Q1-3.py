 # %% [markdown]
# # Step 1
# ## I reviewed and brainstromed for both datasets, but will be working with them one at a time.

# **My thouhgts on the College Completion Data Set:**
#
# Students from different backgrounds graduate at different rates.
#
# Financial need may make it harder for students to complete college.
#
# # Colleges vary in how much financial aid they provide to students.

# **My thouhgts on the Student Placement Dataset:**
#
# Some students get placed after graduation while others do not.
#
# Salary offers vary widely among placed students.
#
# Students have different academic backgrounds and specializations.



# %% [markdown]
# # Step 2 
# ## Question and IBM for College Completion DataSet:
# **Question:** How does student financial need impact college completion rates?
#
# **Independent** Business Metric: Percentage of students receiving Pell Grants
# %%
import sys
print(sys.executable)
# %%
# Imports - Libraries needed for data manipulation and ML preprocessing
import pandas as pd  # For data manipulation and analysis
# %%
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
# %%
# Make sure to install sklearn in your terminal first!
# Use: pip install scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data
# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
# %%from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data


# %%
#read in college completion data
url = "https://raw.githubusercontent.com/AudreyStelle/Data_Cleaning_Lab_I/main/cc_institution_details.csv"
collegecompletion = pd.read_csv(url)
 #%%
 # Let's check the structure of the dataset and see if we have any issues with variable classes
#usually it's converting things to category.
collegecompletion.info()

# %%
# Save the DataFrame as a local CSV so it appears in the VS Code workspace
# This allows the file to be opened and analyzed using Data Wrangler
collegecompletion.to_csv("cc_institution_details.csv", index=False)

# %%
#another way of checking the structure of the dataset. Simpler, but does not give an index
collegecompletion.dtypes
#%%
#Now I observe which columns that will serve my question need to be converted to category

# Resolve column names from indices
cols = collegecompletion.columns[[15, 16, 21]]

# Convert using column names (no view/copy ambiguity)
collegecompletion[cols] = collegecompletion[cols].astype("category")

# Verify
collegecompletion[cols].dtypes
collegecompletion.dtypes 

# %%
## iloc-based assignment did not persist due to pandas view vs copy ambiguity;
# converting by column name ensures the dtype change is retained.
#%%
#Let's take a closer look at pell_value
collegecompletion.pell_value.value_counts() #value_counts() simply displays variable counts as a vertical table.
#%%


# %%
# Create a new column that groups Pell Grant percentages into categories
collegecompletion["pell_group"] = pd.cut(
    collegecompletion["pell_value"],      # numeric Pell Grant percentage
    bins=[0, 33, 66, 100],                 # define Low, Medium, High ranges
    labels=["Low", "Medium", "High"]       # names for each group
)

# %%
#Check that it worked
collegecompletion["pell_group"].dtype

# %%
#Let's take a closer look at pell_group
collegecompletion.pell_group.value_counts() #value_counts() simply displays variable counts as a vertical table.

# %%
#Check if it worked
collegecompletion["grad_150_group"].value_counts()
#%%
collegecompletion.level.value_counts() #looks good, only two groups
# %%
collegecompletion.control.value_counts() #looks good, only three types
#%%
#Nothing to collapse that I can find obvious
collegecompletion.info
# %%
#Checking to see if pell_value is centered
collegecompletion['pell_value'].mean()#not centered

# %%
#Checking to see if pell_value is scaled
collegecompletion['pell_value'].std()


# %%
#Centering and Standardizing Data
pell_value = StandardScaler().fit_transform(collegecompletion[['pell_value']])
#reshapes series into an appropriate argument for the function fit_transform: an array

pell_value[:10] #essentially taking the zscores of the data, here are the first 10 valu

# %%
#Let's look at min-max scaling, placing the numbers between 0 and 1. 
aid_value = MinMaxScaler().fit_transform(collegecompletion[['aid_value']])
aid_value[:10]
# %%

# %%
#Let's check just to be sure the relationships are the same
collegecompletion.pell_value.plot.density()
# %%
pd.DataFrame(pell_value).plot.density() #Checks out!
# %%
# Next let's one-hot encode those categorical variables

category_list = [
    'level',        # 2-year vs 4-year institution
    'control',      # Public / Private not-for-profit / Private for-profit
    'hbcu',         # Historically Black College or University
    'flagship',     # Flagship institution
    'pell_group'    # Low / Medium / High Pell (created earlier)
]

# get_dummies encodes categorical variables into binary by adding
# an indicator column for each group of a category and assigning
# it 0 if false or 1 if true

collegecompletion_1h = pd.get_dummies(
    collegecompletion,
    columns=category_list,
    drop_first=True
)

collegecompletion_1h  # see the difference? This is one-hot encoding!


# %%
#Dropping unnessecary columns
collegecompletion = collegecompletion.drop(
    columns=['exp_award_value ', 'exp_award_state_value', 'exp_award_natl_value ', 'exp_award_percentile', 'fte_value','fte_percentile', 'med_sat_value ','med_sat_percentile ', 'endow_value ', 'endow_percentile', 'vsa_year',
]
)


# %%
#Dropping unnessecary columns
cols_to_drop = [
    'vsa_year',
    'vsa_grad_after4_first',
    'vsa_grad_elsewhere_after4_first',
    'vsa_enroll_after4_first',
    'vsa_enroll_elsewhere_after4_first',
    'vsa_grad_after6_first',
    'vsa_grad_elsewhere_after6_first',
    'vsa_enroll_after6_first',
    'vsa_enroll_elsewhere_after6_first',
    'vsa_grad_after4_transfer',
    'vsa_grad_elsewhere_after4_transfer',
    'vsa_enroll_after4_transfer',
    'vsa_enroll_elsewhere_after4_transfer',
    'vsa_grad_after6_transfer',
    'vsa_grad_elsewhere_after6_transfer',
    'vsa_enroll_after6_transfer',
    'vsa_enroll_elsewhere_after6_transfer'
]

# %%
# Create target variable based on median graduation rate
median_grad = collegecompletion['grad_150_value'].median()

collegecompletion['high_completion'] = (
    collegecompletion['grad_150_value'] >= median_grad
).astype(int)

# Check the result
collegecompletion['high_completion'].value_counts()

# %%
# Visualize the distribution of six-year graduation rates
collegecompletion.boxplot(
    column='grad_150_value',
    vert=False,
    grid=False
)

# %%
# View summary statistics for graduation rates
# We will use the 75th percentile as the cutoff for the positive class
collegecompletion['grad_150_value'].describe()


# %%
# Calculate the 75th percentile (top quartile) of graduation rates
q75 = collegecompletion['grad_150_value'].quantile(0.75)

# %%
# Create a binary target variable
# 1 = institution is in the top quartile of graduation rates
# 0 = institution is below the top quartile
collegecompletion['high_completion_q'] = (
    collegecompletion['grad_150_value'] >= q75
).astype(int)

# %%
# Count the number of institutions in each class
collegecompletion['high_completion_q'].value_counts()

# %%
[markdown]
# The continuous graduation rate variable was converted into a binary target by classifying institutions in the top quartile of six-year graduation rates as the positive class.
# %%
# Recreate the target variable on the encoded dataset
collegecompletion_1h['high_completion_q'] = collegecompletion['high_completion_q']
#%%
# Now we partition the data

Train, Test = train_test_split(
    collegecompletion_1h,
    train_size=0.55,
    stratify=collegecompletion_1h.high_completion_q,
    random_state=42
)

# stratify preserves the proportion of institutions with
# high vs low graduation rates when splitting the data


# %%
[markdown]
# The dataset was partitioned into training and testing sets using stratified sampling to preserve the proportion of high-completion institutions.

# %%
[markdown]
# # Now to do step two again with the Job_Placement Data Set
# %%
[markdown]
# I will read in the 2nd data set
#%%
import pandas as pd
url = "https://raw.githubusercontent.com/AudreyStelle/Data_Cleaning_Lab_I/main/Placement_Data_Full_Class.csv"
Placement_Data_Full_Class = pd.read_csv(url)


#%%
# Let's check the structure of the dataset and see if we have any issues with variable classes
#usually it's converting things to category.
Placement_Data_Full_Class.info()
#%% 
[markdown]
# # Step 2 
# ## Question and IBM for Job Placement DataSet:
# **Question:** How does academic performance influence employment outcomes for MBA students?
#
# **Independent** Business Metric: MBA Percentage (mba_p)
# %%
# Save the DataFrame as a local CSV so it appears in the VS Code workspace
# This allows the file to be opened and analyzed using Data Wrangler
Placement_Data_Full_Class.to_csv("cc_institution_details.csv", index=False)
#%%
Placement_Data_Full_Class.to_csv("Placement_Data_Full_Class.csv", index=False)

# %%
#another way of checking the structure of the dataset. Simpler, but does not give an index
Placement_Data_Full_Class.dtypes

# %%
# This variable is qualitative (labels, not numbers)
# Therefore, it should be treated as a categorical variable

Placement_Data_Full_Class["ssc_b"] = (
    Placement_Data_Full_Class["ssc_b"].astype("category")
)


# %%
#Check to see if the conversion worked
Placement_Data_Full_Class.dtypes

# %%
# This column is a qualitative variable and should be treated as categorical

Placement_Data_Full_Class["hsc_b"] = (
    Placement_Data_Full_Class["hsc_b"].astype("category")
)


# %%
# This column is a qualitative variable and should be treated as categorical

Placement_Data_Full_Class["hsc_s"] = (
    Placement_Data_Full_Class["hsc_s"].astype("category")
)
#%%
# This column contains qualitative labels and not numerical values and shoudl be converted to categorical
Placement_Data_Full_Class["degree_t"] = (
    Placement_Data_Full_Class["degree_t"].astype("category")
)

# %%
#Check to see if ALL the conversion worked
Placement_Data_Full_Class.dtypes

# %%
# The Salary column has 63 missing values, but the missing vaues corelate with the 'status' column 
# The not placed values do not have any salarly values
# Because of the recognized pattern, I will keep the Salarly column but filter it to only placed students
placed_df = Placement_Data_Full_Class[
    Placement_Data_Full_Class["status"] == "Placed"
]

# %%
# After lookiing at my question and the data, i do not need to colapse vector levels
#%%
#Centering and Standardizing Data
sl_no = StandardScaler().fit_transform(Placement_Data_Full_Class[['sl_no']])
#reshapes series into an appropriate argument for the function fit_transform: an array

sl_no[:10] #essentially taking the zscores of the data, here are the first 10 values



# %%
#Let's look at min-max scaling, placing the numbers between 0 and 1. 
sl_no = MinMaxScaler().fit_transform(Placement_Data_Full_Class[['sl_no']])
sl_no[:10]
# %%
#Let's check just to be sure the relationships are the same
Placement_Data_Full_Class.sl_no.plot.density()
# %%

pd.DataFrame(sl_no).plot.density() #Checks out!
# %%
#Now we can move forward in normalizing the numeric values and create a index based on numeric columns:
abc = list(Placement_Data_Full_Class.select_dtypes('number')) #select function to find the numeric variables and create a list  

Placement_Data_Full_Class[abc] = MinMaxScaler().fit_transform(Placement_Data_Full_Class[abc])
Placement_Data_Full_Class #notice the difference in the range of values for the numeric variables

# %%# Next let's one-hot encode those categorical variables

category_list = list(Placement_Data_Full_Class.select_dtypes('category')) #select function to find the categorical variables and create a list  

Placement_Data_Full_Class_1h = pd.get_dummies(Placement_Data_Full_Class, columns = category_list) 
#get_dummies encodes categorical variables into binary by adding in indicator column for each group of a category 
#and assigning it 0 if false or 1 if true
Placement_Data_Full_Class_1h #see the difference? This is one-hot encoding!
# %%
# Because the Data set had few columns, I did not want to minimize the data anymore and will not drop any columns
# %%
# Inspect target distribution
Placement_Data_Full_Class['status'].value_counts(normalize=True)

# Convert target to Boolean for classification
Placement_Data_Full_Class['placed_bool'] = (
 Placement_Data_Full_Class['status'] == 'Placed'
)

# %%
# Visualize a continuous variable by placement outcome
# This mirrors the cereal rating boxplot step

Placement_Data_Full_Class.boxplot(
    column='salary',
    by='STATUS',
    vert=False,
    grid=False
)

# %%
# Add a binary predictor based on a continuous variable (do NOT replace original)
# This mirrors how rating_f was created from rating in the cereal example

mba_cutoff = Placement_Data_Full_Class['mba_p'].quantile(0.75)

Placement_Data_Full_Class['MBA_HIGH'] = pd.cut(
    Placement_Data_Full_Class['mba_p'],
    bins=[-1, mba_cutoff, 100],
    labels=[0, 1]
)

Placement_Data_Full_Class
# Notice the new column MBA_HIGH:
# it is now binary based on whether mba_p is in the top quartile or not

# %%
# Check the prevalence (baseline rate)
prevalence = (
    Placement_Data_Full_Class['MBA_HIGH'].value_counts()[1]
    / len(Placement_Data_Full_Class['MBA_HIGH'])
)
prevalence  # gives the proportion of students with high MBA scores (baseline)

# %%
from sklearn.model_selection import train_test_split

# Define target and predictors
y = Placement_Data_Full_Class['placed_bool']
X = Placement_Data_Full_Class.drop(columns=['status', 'placed_bool'])

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# %%
# Compare prevalence in full data vs splits
y.mean(), y_train.mean(), y_test.mean()


# %%
[markdown]
# # Step 3
# The College Completion Dataset adressed the problem I stated becuase it includes graduation rates and financial aid indicators, making it good for analyzing hwo student financial need relates to college completion. However, I am worried about how the data is summarized at the school level, has some missing values, and does not include all factors that may affect graduation rates.
# The Job Placement Dataset adressed the problem I presented because the dataset contains performance measures and placement outcomes. This is good because it allows an analysis of how academic acheivment affects job placement. I do have some concerns that revolve around the fact that salary data is missing for unplaced students and the sample size is smaller than I would prefer.