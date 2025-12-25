import pandas as pd
import os

# Create a sample DataFrame with column names
data = {'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
    }

df = pd.DataFrame(data)

# Adding new row to df for V2
new_row_loc = pd.Series(
    {'Name': 'Roman', 'Age': 24, 'City': 'Dhaka'},
    index=df.columns
)

df.loc[len(df)] = new_row_loc


# Ensure the "data" directory exists
data_dir = 'DVC/data'
os.makedirs(data_dir, exist_ok=True)

# Define the file path
file_path = os.path.join(data_dir, 'sample_data.csv')

# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)

print(f"CSV file saved to {file_path}")

# pip install dvc
# Now we do "dvc init" (creates .dvcignore, .dvc)
# Now do "mkdir DVC/S3" (creates a new S3 directory)
# Now we do "dvc remote add -d s3remote DVC/S3"
# Next "dvc add DVC/data/" 
# Now it will ask to do: ("git rm -r --cached 'DVC/data'" and "git commit -m "stop tracking DVC\data"")
# Because initially we were tracking data/ folder from git so now we remove it for DVC to handle.

# Again we do "dvc add DVC/data/" (creates DVC/data.dvc)
# Now - "dvc commit" and then "dvc push"
# Do a git add-commit-push to mark this stage as first version of data.

# Now make changes to main.py to append a new row in data, check changes via "dvc status"
# Again - - "dvc commit" and then "dvc push"
# Then git add-commit-push (we're saving V2 of our data at this point)
# Check dvc/git status, everything should be upto date.

# "git checkout hash" to revert to any version of the code
# "dvc status" will show changed because the code version and the data version doesn't match
# "dvc pull" will pull the corresponding data version