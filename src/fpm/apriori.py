

import pandas as pd
from apyori import apriori

data_file = "../../resources/data/store_data.csv"
raw_data = pd.read_csv(data_file, header=None)
print(raw_data)

data_rows = []
number_of_rows = 22
for i in range(0, number_of_rows):
    data_rows.append([str(raw_data.values[i, j]) for j in range(0, 6)])

for i in range(0, number_of_rows):
    print(data_rows[i])

association_rules = apriori(data_rows, min_support=0.1, min_confidence=1.0,
                            min_lift=3.0, min_length=2)
association_results = list(association_rules)
print(len(association_results))
for i in range(len(association_results)):
    print(association_results[i])

