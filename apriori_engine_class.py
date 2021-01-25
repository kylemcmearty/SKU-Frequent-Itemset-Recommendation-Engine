"""
Generates "recommendations" for a list of unique identifiers (itemcodes) \
based on frequently purchased itemsets. This program uses the mlxtend package \
and the Apriori algorithm to mine assocation patterns between frequently \
purchased items and creates rules for itemcode recommendations \
based on the defined parameters.

@AprioriAuthor: @rasbt
https://github.com/rasbt/mlxtend/blob/master/mlxtend/frequent_patterns/apriori.py
@mlxtend.version: 0.18.0
@Python.version: 3.9.1
@Flake8: enabled

:raises AttributeError: validate df contains itemcode and transaction columns
:return: rules of assc rules that match the user itemcode input
:rtype: pd.DataFrame
"""

from datetime import datetime as dt

from mlxtend.frequent_patterns import (
    apriori,
    association_rules,
)
from mlxtend.preprocessing import TransactionEncoder

import pandas as pd


def read_csv(filename: str):
    """Load dataset into dataframe."""
    return pd.read_csv(filename, dtype='object')


@pd.api.extensions.register_dataframe_accessor("mba")
class MarketBasketAnalysis:
    """Market Basket Analysis with the mlxtend Apriori alogrithm module."""

    def __init__(self, pandas_obj):
        """Initialize objects."""
        self._validate(pandas_obj)
        self.df = pandas_obj

    @staticmethod
    def _validate(obj):
        """Verify there is a column transaction_id and a column item_code."""
        if ("transaction_id" not in obj.columns
           or "item_code" not in obj.columns):
            raise AttributeError("Dataset must contain columns"
                                 "'transaction_id' and 'item_code'.")

    def prepare_dataset(self) -> list[str]:
        """
        Create a list of lists contatining unique itemcodes in every tranxtion.

        :return: a list of lists containing all the unique itemcode
        :rtype: List
        """
        # Groupby transaction_ID and count the number of items
        # sold by transaction group
        self.df['frequency'] = self.df.groupby('transaction_id'
                                               )['transaction_id'
                                                 ].transform('count')

        group_result = self.df[['transaction_id', 'frequency', 'item_code']]

        # Group_result by transaction id and
        # aggregate unique item_codes together
        id_list = group_result.groupby('transaction_id', sort=False
                                       ).item_code \
                                        .unique() \
                                        .agg(','.join) \
                                        .reset_index()

        item_code_list = id_list["item_code"].tolist()

        return [str.split(x, sep=',') for x in item_code_list]

    def calculate_itemset_frequency(self,
                                    transaction_list: list[str],
                                    itemcode_input: str,
                                    min_sup: float = 0.0003,
                                    min_thrsh: float = 0.02):
        """
        Calculate frequent itemsets \
        with the mlxtend apriori algorithm implementation.

        :param transaction_list: list of lists of itemcode transactions
        :type transaction_list: list
        :param itemcode_input: the itemcode to find matching rules
        :type itemcode_input: string
        :param min_sup: calculated freq support for Apriori, defaults to 0.0003
        :type min_sup: float, optional
        :param min_thrsh: minimum threshold for assc rules, defaults to 0.02
        :type min_thrsh: float, optional
        """
        # Apriori Frequent Itemsets One-Hot Encoding
        te = TransactionEncoder()
        te_ary = te.fit(transaction_list).transform(transaction_list)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # Set min_support level and low_memory = True for large datasets
        # count legth of itemsets
        frequent_itemsets = apriori(df,
                                    min_support=min_sup,
                                    use_colnames=True,
                                    low_memory=True)

        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(
                                                          lambda x: len(x))
        # Return only pairs or triplets frequent items
        itemset_df = frequent_itemsets[(frequent_itemsets['length'] <= 3)]

        self.itemcode_rcmds(itemset_df, itemcode_input, min_thrsh)

    def itemcode_rcmds(self,
                       dataset,
                       itemcode_input: str,
                       min_thrsh: float) -> pd.core.frame.DataFrame:
        """
        Set Association Rules for our frequent itemset recommendations.

        :param dataset: the frequent itemset dataframe
        :type dataset: pd.DataFrame
        :param itemcode_input: the itemcode input to find matching assc rules
        :type itemcode_input: String
        :param min_thrsh: minimum threshold for assc rules
        :type min_thrsh: Float
        :return: returns the assc rules that match the itemcode input
        :rtype: pd.DataFrame
        """
        # Calculate confidence of associations and set a minimum
        # threshold on returned predictions
        start_time = dt.now()
        Assc_rules = association_rules(dataset,
                                       metric='confidence',
                                       min_threshold=min_thrsh)
        Assc_rules = Assc_rules.sort_values(by='confidence', ascending=False)
        end_time = dt.now() - start_time

        results = Assc_rules[Assc_rules.antecedents.map(
                                lambda A: A.issubset(itemcode_input))]

        if results.shape[0] < 1:
            return print(f"\nno rules found that match {itemcode_input=}\n")

        else:
            # Rearrange dataframe
            reorder_columns = ['antecedents', 'consequents',
                               'consequent support', 'confidence']
            results = results[reorder_columns].reset_index()

            print(
                f"\n\n    List of itemcodes: {itemcode_input}\n"
                f"\n\t*** {results.shape[0]} assc rules matching "
                "the input itemcodes ***\n"
                f"\n{results}\n"
                f"\nAssociation rules calculated in {end_time}\n"
            )

            return results


if __name__ == '__main__':

    # Get itemcode input from user
    itemcode_input = list(map(str,
                              input("\nEnter multiple itemcodes"
                                    "(with whitespace inbetween): ").split()))

    # load dataset from csv or from database
    csv_dataset = './Finished_dataset.csv'
    df = read_csv(csv_dataset)

    data = df.mba.prepare_dataset()
    df.mba.calculate_itemset_frequency(data, itemcode_input,
                                       min_sup=0.001, min_thrsh=0.02)
