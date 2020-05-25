
"""
Author: Kyle A. McMearty - 05/2020
Generates "recommendations" for a list of unique identifiers (SKUs) based on frequently purchased itemsets.
This program uses the mlxtend package and the Apriori algorithm to mine assocation patterns between
frequently purchased items and creates rules for SKU recommendations based on the defined parameters.

Apriori Source: @rasbt - https://github.com/rasbt/mlxtend/blob/master/mlxtend/frequent_patterns/apriori.py
"""


# Import dependencies
import pandas as pd
import numpy as np
import mlxtend
import warnings; warnings.simplefilter('ignore')
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def sku_lists():
    """
    Returns a dataset of transactions as a list of lists to pass into the frequent itemsets function

    """
 
    # Read filename
    df = pd.read_csv('./Finished_dataset.csv')

    # Rename columns that match df column names
    df['trnsact_id'] = df['id']
    
    # Groupby Transaction_ID and count the number of SKUs sold by transaction group
    df['frequency'] = df.groupby('trnsact_id')['trnsact_id'].transform('count')
    result = df[['trnsact_id', 'frequency', 'sku']]
    result['sku'] = result['sku'].astype(str)

    # Group result by transaction id and aggregate unique sku numbers together
    id_list = result.groupby('trnsact_id',sort=False).sku.unique().agg(','.join).reset_index()
    sku_list = id_list["sku"].tolist()

    # Split sku_list by comma which creates a list of lists
    dataset = [str.split(x, sep=',') for x in sku_list]

    return dataset


def freq_rules(passed_dataset, min_sup=0.0003):
    """
    Calculates frequent itemsets with the apriori algorithm.

    Keyword Arguments:
     - passed_dataset: the list of lists from our intial dataset that we want process into frequent itemsets
     - min_sup: sets the minimum support for associations between frequent itemsets

    """

    # Apriori Frequent Itemsets One-Hot Encoding 
    te = TransactionEncoder()
    te_ary = te.fit(passed_dataset).transform(passed_dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Set min_support level and low_memory = True for large datasets, count legth of itemsets
    frequent_itemsets = apriori(df, min_support=min_sup, use_colnames=True, low_memory=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

    # Set filter on length of frequent items to only pairs or triplets
    dataset = frequent_itemsets[(frequent_itemsets['length'] <= 3)]

    return dataset


def sku_rcmds(dataset, sku_input, min_thrsh=0.01):
    """
    Set Association Rules for our frequent itemset recommendations.

    Keyword Arguments:
     - dataset: list of calculations for our frequent items dataset
     - sku_input: list of sku numbers passed in by the user
     - min_thrsh: sets the bottom threshold of confidence to return a recommendation

    """

    # Calculate confidence of associations and set a minimum threshold on returned predictions
    Assc_rules = association_rules(dataset, metric='confidence', min_threshold=min_thrsh)
    Assc_rules = Assc_rules.sort_values(by='confidence', ascending=False)

    # List of SKUs
    results = Assc_rules[Assc_rules.antecedents.map(lambda A: A.issubset(sku_input))]

    # Rearrange dataframe
    reord_colm = ['antecedents', 'consequents', 'consequent support', 'confidence']
    results = results[reord_colm].reset_index()
    
    # Format print recommendations
    print(f'')
    print(f"\n    List of SKUs: {sku_input}\n")
    print(f'\n    *** {results.shape[0]} total recommendations made for List of SKUs ***\n')
    print(f'\n{results}\n')

    return results


def main():
    """
    Test SKUs
    1985100 1986205 1986161 2029905 1955415 1986189 2042524
    single: 1985093

    """

    sku_listolists = sku_lists()
    freq_items = freq_rules(sku_listolists, min_sup=0.0005)
    sku_input = list(map(str, input(f"\nEnter multiple SKUs (with whitespace inbetween): ").split()))
    sku_rcmds(freq_items, sku_input, min_thrsh=0.02)

    pass


if __name__ == '__main__':
    main()
