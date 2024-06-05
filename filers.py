import pandas as pd
import numpy as np

# Sample DataFrame
df = pd.DataFrame({'col': [np.nan, 4, np.nan, 4, np.nan]})

# Replace NaN with 0 and 4 with 1
df['col'] = df['col'].replace({np.nan: 0, 4: 1})

# Display the DataFrame
print(df)