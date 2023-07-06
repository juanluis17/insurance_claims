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

            "precision_macro": ["mean", "std"],
            "recall_macro": ["mean", "std"],
            "f1_macro": ["mean", "std"],

            "precision_car": ["mean", "std"],
            "precision_home": ["mean", "std"],
            "precision_life": ["mean", "std"],
            "precision_health": ["mean", "std"],
            "precision_sports": ["mean", "std"],

            "recall_car": ["mean", "std"],
            "recall_home": ["mean", "std"],
            "recall_life": ["mean", "std"],
            "recall_health": ["mean", "std"],
            "recall_sports": ["mean", "std"],

            "f1_car": ["mean", "std"],
            "f1_home": ["mean", "std"],
            "f1_life": ["mean", "std"],
            "f1_health": ["mean", "std"],
            "f1_sports": ["mean", "std"],

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
