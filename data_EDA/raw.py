import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_style('darkgrid')
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.facecolor'] = '#00000000'

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_risk_dataset.csv")
raw_data = pd.read_csv(csv_path)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Информация изначального датасете
if __name__ == "__main__":
    print(f'{raw_data.info()}\n\n')
    print(f'{raw_data.isnull().sum()}\n\n')
    print(f'{raw_data.nunique()}\n\n')
    print(f'{raw_data.loan_status.value_counts()}\n\n')
    print(f'{raw_data.describe()}\n\n')
    print(f'{raw_data.head()}\n\n')
    print(f'{raw_data.tail()}')

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(raw_data.corr(numeric_only=True), vmax=.8, square=True, annot=True, cmap='Greens', ax=ax)
    plt.title('Correlation', fontsize=15)
    plt.show()