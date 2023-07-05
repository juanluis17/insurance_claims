import pandas as pd


def group_data(data, criteria):
    """
    Groups the data based on the specified criteria and calculates the mean, standard deviation, and variance
    of various metrics.

    Args:
        data (pandas.DataFrame): The data to group and analyze.
        criteria (list): The column(s) to group the data by.

    Returns:
        pandas.DataFrame: The grouped data with mean, standard deviation, and variance of different metrics.

    """
    grouped_multiple = data.groupby(criteria).agg(
        {
            "accuracy": ["mean", "std"],

            "TP": ["mean", "std"],
            "FN": ["mean", "std"],
            "FP": ["mean", "std"],
            "TN": ["mean", "std"],

            "precision_macro": ["mean", "std"],
            "recall_macro": ["mean", "std"],
            "f1_macro": ["mean", "std"],
            "f1_weighted": ["mean", "std"],
            "mcc": ["mean", "std"],
            "roc_au_score_macro": ["mean", "std"],
            "precision_positive": ["mean", "std"],
            "recall_positive": ["mean", "std"],
            "f1_positive": ["mean", "std"],
            "precision_negative": ["mean", "std"],
            "recall_negative": ["mean", "std"],
            "f1_negative": ["mean", "std"]
        })
    grouped_multiple = grouped_multiple.reset_index()
    return grouped_multiple


def export_to_xlsx(data, save_dir):
    """
    Export a DataFrame to an Excel file.

    Args:
        data (pandas.DataFrame): The DataFrame to export.
        save_dir (str): The directory and filename to save the Excel file.

    Returns:
        None

    """
    writer = pd.ExcelWriter(save_dir, engine="xlsxwriter")
    data.to_excel(writer, sheet_name="Data", startrow=0, startcol=0, header=True, index=True)
    writer.close()
