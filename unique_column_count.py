import pandas as pd

# Load the CSV file
file_name = 'CSE-CIC-IDS2018/03-02-2018.csv'
df = pd.read_csv(file_name)

# Replace 'ColumnName' with the actual name of the column you're interested in
unique_values = df['Label'].unique()

# Save the unique values to a text file
with open('label_types.txt', 'a') as f:
    f.write(file_name+str(unique_values) + '\n')

# df_filtered = df[df['Label'] != 'Label']
# df_filtered.to_csv('CSE-CIC-IDS2018/test.csv', index=False)

