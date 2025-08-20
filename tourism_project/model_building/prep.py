# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/sasipriyank/tourismlops/tourism.csv"
tour_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
tour_dataset.drop(columns=['CustomerID'], inplace=True)

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',               # Customer's age
    'CityTier',           # city category based on development, population, and living standards((tier1,tier2..)
    'DurationOfPitch',     # Duration of the sales pitch delivered to the customer.
    'NumberOfPersonVisiting', # Total number of people accompanying the customer on the trip.
    'NumberOfFollowups',    # Total number of follow-ups by the salesperson after the sales pitch.
    'PreferredPropertyStar',    # Customer’s estimated salary
    'PitchSatisfactionScore', #Score indicating the customer's satisfaction with the sales pitch.
    'NumberOfTrips',    # Average number of trips the customer takes annually.
    'Passport',    # Whether the customer holds a valid passport (0: No, 1: Yes)
    'OwnCar',    # Whether the customer owns a car (0: No, 1: Yes)
    'NumberOfChildrenVisiting',# Number of children below age 5 accompanying the customer.
    'MonthlyIncome' #Gross monthly income of the customer.


]

# List of categorical features in the dataset
categorical_features = [
    'Occupation',         # Customer Occupation (e.g Salaried)
    'TypeofContact', # type of contact
    'Gender', #Gender of the customer (Male, Female).
    'ProductPitched', # The type of product pitched to the customer
    'MaritalStatus', # Marital status of the customer (Single, Married, Divorced).
    'Designation' # Customer's Designation (e.g., Manager, VP).

]

# Define predictor matrix (X) using selected numeric and categorical features
X = tour_dataset[numeric_features + categorical_features]

# Define target variable
y = tour_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="sasipriyank/tourismlops",
        repo_type="dataset",
    )
