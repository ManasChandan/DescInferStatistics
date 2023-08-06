import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scipy.stats as scipy_stats

def make_ordinal_column(data: pd.DataFrame, cont_column: str, new_col_name: str = "", list_of_categories: list = []):
    if data.shape[0] == 0:
        raise Exception("There is no data -- Please check")
    if cont_column not in data.columns:
        raise Exception("The Column is not in the data")
    print(cont_column)
    if new_col_name == "":
        new_col_name = f"{cont_column}_ordinal"
    print(new_col_name)
    if len(list_of_categories) == 0:
        raise Exception("The list_of_categories is empty not acceptable")
    print(list_of_categories)
    data[new_col_name] = pd.cut(data[cont_column], bins=list_of_categories, labels=[
        i for i in range(len(list_of_categories)-1)
    ], ordered=False, include_lowest=True)
    data[new_col_name] = data[new_col_name].astype(str)
    if data[new_col_name].isna().sum() != 0:
        print("ERROR WARNING: There is some anomoly in the data. Kindly Check. Currently replacing with mode of the column")
        data[new_col_name].fillna(data[new_col_name].value_counts().index[0]) # type: ignore
    print("Successfully made the new ordinal column")
    return data

def point_estimation_impute(data: pd.DataFrame, col_name: str, metric_name: str = "mean", metric_value: object = None):
    if data.shape[0] == 0:
        raise Exception("There is no data -- Please check")
    if col_name not in data.columns:
        raise Exception("The Column is not in the data")
    print(col_name)
    if metric_value is None:
        if metric_name == "mean":
            metric_value = np.nanmean(data[col_name])
        elif metric_name == "median":
            metric_value = np.nanmedian(data[col_name])
        elif metric_name == "mode":
            metric_value = data[col_name].value_counts().index[0]        
    data[col_name] = data[col_name].fillna(metric_value) # type: ignore
    return data

def shapiro_wilk_test(data: pd.DataFrame, col_name: str, significance_level: float = 0.05):
    if data.shape[0] == 0:
        raise Exception("There is no data -- Please check")
    if col_name not in data.columns:
        raise Exception("The Column is not in the data")
    if data[col_name].isna().sum() != 0:
        raise Exception("There Is a null value in the data. Please check")
    print("Perfoming shapiro wilk test")
    shapiro_stat, shapiro_p_value = scipy_stats.shapiro(data[col_name].values)
    print(f"The Shapiro Wilk Statistic Value is {shapiro_stat}")
    print(f"The P value is {shapiro_p_value}")
    return shapiro_p_value > significance_level

def is_normal_distribution(data: pd.DataFrame, col_name: str, significance_level: float = 0.05):
    if shapiro_wilk_test(data, col_name, significance_level):
        print("Failed to reject H0: Data is normally distributed")
        return True
    else:
        print("Rejected H0: Data is not normally distributed")
        return False

def generate_sample_statistics(data: pd.DataFrame, col_name: str, significance_level: float = 0.05):
    if data.shape[0] == 0:
        raise Exception("There is no data -- Please check")
    if col_name not in data.columns:
        raise Exception("The Column is not in the data")
    if data[col_name].isna().sum() != 0:
        raise Exception("There Is a null value in the data. Please check")
    is_normally_distrbuted = is_normal_distribution(data, col_name, significance_level)
    mean = data[col_name].mean()
    median = data[col_name].median()
    std_dev = data[col_name].std()
    std_error = data[col_name].sem()
    confidence_level = 1 - significance_level
    coefficient_of_variation = (std_dev / mean) * 100
    pearsons_second_coefficient = scipy_stats.skew(data[col_name])
    kurtosis = scipy_stats.kurtosis(data[col_name])
    excess_kurtosis = kurtosis - 3
    q1 = np.percentile(data[col_name], 25)
    q2 = np.percentile(data[col_name], 50)  # Median
    q3 = np.percentile(data[col_name], 75)
    mean_confidence_interval = scipy_stats.t.interval(
        alpha=confidence_level,
        df=data[col_name].shape[0] - 1,
        loc=mean,
        scale=std_error
    )
    median_confidence_interval = scipy_stats.t.interval(
        alpha=confidence_level,
        df=data[col_name].shape[0] - 1,
        loc=median,
        scale=std_error
    )
    return {
        "Normal Dist.": is_normally_distrbuted,
        "mean": mean, 
        "median": median,
        "std_dev": std_dev,
        "std_error": std_error,
        "Coefficient of Variation": coefficient_of_variation,
        "Mean CI-Lower": mean_confidence_interval[0],
        "Mean CI-Upper": mean_confidence_interval[1],
        "Median CI-Lower": median_confidence_interval[0],
        "Median CI-Upper": median_confidence_interval[1],
        "Pearsons Second Coeff": pearsons_second_coefficient, 
        "Kurtosis": kurtosis,
        "Excess Kurtosis": excess_kurtosis,
        "Q1": q1,
        "Q2": q2, 
        "Q3": q3
    }

def map_z_and_modified_z(data: pd.DataFrame, col_name: str):
    if data.shape[0] == 0:
        raise Exception("There is no data -- Please check")
    if col_name not in data.columns:
        raise Exception("The Column is not in the data")
    if data[col_name].isna().sum() != 0:
        raise Exception("There Is a null value in the data. Please check")
    mad = scipy_stats.median_abs_deviation(data[col_name])
    mean = data[col_name].mean()
    median = data[col_name].median()
    std_dev = data[col_name].std()
    data[f"{col_name}_Z_Score"] = (data[col_name]-mean)/std_dev
    data[f"{col_name}_Mod_Z_Score"] = 0.6745 * (data[col_name] - median) / mad
    return data

def two_sample_t_test(data: pd.DataFrame, col1: str, col2: str, significance_value: float = 0.05):
    if data.shape[0] == 0:
        raise Exception("There is no data -- Please check")
    if col1 not in data.columns:
        raise Exception(f"The Column {col1} is not in the data")
    if data[col1].isna().sum() != 0:
        raise Exception(f"There Is a null value in the data {col1}. Please check")
    if col2 not in data.columns:
        raise Exception(f"The Column {col2} is not in the data")
    if data[col2].isna().sum() != 0:
        raise Exception(f"There Is a null value in the data {col2}. Please check")
    t_statistic, p_value = scipy_stats.ttest_ind(data[col1].values, data[col2].values)
    print(f"The value of T-Stat is = {t_statistic}")
    print(p_value)
    return "Reject Ho: The Mean is significantly different" if p_value < significance_value else "Failed To reject Ho: Means are same"

def two_sample_f_test(data: pd.DataFrame, col1: str, col2: str, significance_value: float = 0.05):
    if data.shape[0] == 0:
        raise Exception("There is no data -- Please check")
    if col1 not in data.columns:
        raise Exception(f"The Column {col1} is not in the data")
    if data[col1].isna().sum() != 0:
        raise Exception(f"There Is a null value in the data {col1}. Please check")
    if col2 not in data.columns:
        raise Exception(f"The Column {col2} is not in the data")
    if data[col2].isna().sum() != 0:
        raise Exception(f"There Is a null value in the data {col2}. Please check")
    variance_ratio = np.var(data[col1], ddof=1) / np.var(data[col2], ddof=1)
    p_value = scipy_stats.f.cdf(variance_ratio, data[col1].shape[0]-1, data[col2].shape[0]-1)
    print(p_value)
    return "Reject Ho: The Variance is significantly different" if p_value < significance_value else "Failed To reject Ho: Variances are same"

def cosine_similarity(data: pd.DataFrame, col1: str, col2: str):
    if data.shape[0] == 0:
        raise Exception("There is no data -- Please check")
    if col1 not in data.columns:
        raise Exception(f"The Column {col1} is not in the data")
    if data[col1].isna().sum() != 0:
        raise Exception(f"There Is a null value in the data {col1}. Please check")
    if col2 not in data.columns:
        raise Exception(f"The Column {col2} is not in the data")
    if data[col2].isna().sum() != 0:
        raise Exception(f"There Is a null value in the data {col2}. Please check")
    cosine_similarity = np.dot(np.array(data[col1].values), np.array(data[col2].values)) / (np.linalg.norm(data[col1]) * np.linalg.norm(data[col2]))
    cosine_distance = 1 - cosine_similarity
    return cosine_similarity, cosine_distance

def anova_test(data: pd.DataFrame, categorical_col: str, continuous_column: str, significance_value: float = 0.05):
    if data.shape[0] == 0:
        raise Exception("There is no data -- Please check")
    if categorical_col not in data.columns:
        raise Exception(f"The Column {categorical_col} is not in the data")
    if data[categorical_col].isna().sum() != 0:
        raise Exception(f"There Is a null value in the data {categorical_col}. Please check")
    if continuous_column not in data.columns:
        raise Exception(f"The Column {continuous_column} is not in the data")
    if data[continuous_column].isna().sum() != 0:
        raise Exception(f"There Is a null value in the data {continuous_column}. Please check")
    various_grouped_data = list()
    for _, dt in data.groupby(categorical_col):
        various_grouped_data.append(dt[continuous_column])
    f_statistic, p_value = scipy_stats.f_oneway(*various_grouped_data)
    print(f"The Value of F-stat is {f_statistic}")
    return "Reject Ho: The Mean is significantly different" if p_value < significance_value else "Failed To reject Ho: Means are same"

def chi_square_test(data: pd.DataFrame, col1: str, col2: str, significance_value: float = 0.05):
    if data.shape[0] == 0:
        raise Exception("There is no data -- Please check")
    if col1 not in data.columns:
        raise Exception(f"The Column {col1} is not in the data")
    if data[col1].isna().sum() != 0:
        raise Exception(f"There Is a null value in the data {col1}. Please check")
    if col2 not in data.columns:
        raise Exception(f"The Column {col2} is not in the data")
    if data[col2].isna().sum() != 0:
        raise Exception(f"There Is a null value in the data {col2}. Please check")
    contingency_table = pd.crosstab(data[col1], data[col2])
    chi2_statistic, p_value, degrees_of_freedom, expected = scipy_stats.chi2_contingency(contingency_table)
    print(f"The Stats: statistic={chi2_statistic}, p-value={p_value}, dof={degrees_of_freedom}")
    return "Reject Ho: The distribution is significantly different" if p_value < significance_value else "Failed To reject Ho: distribution are same"