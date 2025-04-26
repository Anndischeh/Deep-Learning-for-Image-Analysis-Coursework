import pandas as pd
import json
import matplotlib.pyplot as plt
from Utils.utils import complete_path
from Utils.config import root


class Categories:
    """
    Manages categories, their frequencies, and associated colors.
    """

    def __init__(self, dataset_path, C=20):
        """
        Initializes the Categories object.

        Args:
            dataset_path (str): Path to the COCO dataset.
            C (int, optional): Number of top categories to consider. Defaults to 20.
        """
        with open(complete_path(dataset_path) + 'annotations/instances_train2017.json', 'r') as f:
            temp = json.load(f)
            temp_df = pd.DataFrame(temp['categories']).loc[:, ['id', 'name']].set_index('id')
            self.frequencies = pd.DataFrame(temp['annotations']).loc[:, ['category_id']].groupby(
                'category_id')['category_id'].count().sort_values(ascending=False)
            del temp

        self.C = C
        self.df = temp_df.loc[self.frequencies.iloc[:self.C].index.to_numpy(), :].reset_index()
        self.category_dict = self.df.loc[:, 'name'].to_dict()

        cm = plt.get_cmap('gist_rainbow')
        cNorm = plt.Normalize(vmin=0, vmax=self.C - 1)
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cm)
        self.category_colors = {
            i: (lambda a, b, c, _: (int(a * 255), int(b * 255), int(c * 255)))(*scalarMap.to_rgba(i)) for i in
            range(self.C)}

    def __len__(self):
        return self.C

    def frequency_plot(self, savefig=True):
        """
        Generates and displays a bar plot of category frequencies.

        Args:
            savefig (bool, optional): Whether to save the plot to a file. Defaults to False.
        """
        plt.figure(figsize=(10, 8))
        plt.bar(self.df['name'], self.frequencies.iloc[:self.C] / self.frequencies.sum() * 100,
                color=list(map(lambda x: (x[0] / 255, x[1] / 255, x[2] / 255), self.category_colors.values())))
        plt.xticks(rotation=60)
        plt.title(f"Frequency of occurence of the {self.C} most common categories in the dataset")
        plt.yscale('log')
        if savefig:
            plt.savefig(f"{root}/plots/categories_frequencies_{self.C}.png")
        plt.show()