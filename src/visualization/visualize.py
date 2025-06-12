import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

mpl.style.use("fivethirtyeight")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

bands = df.columns[1:6]

# Melt 5 frequency bands into a single variable
df_melted = df.melt(id_vars="Group", value_vars=bands)
df_melted["variable"] = df_melted["variable"].replace(
    {
        "delta_rbp": "delta",
        "theta_rbp": "theta",
        "alpha_rbp": "alpha",
        "beta_rbp": "beta",
        "gamma_rbp": "gamma",
    }
)
df_melted["Group"] = df_melted["Group"].replace({"A": "AD", "F": "FTD", "C": "CN"})

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_melted,
    x="variable",
    y="value",
    hue="Group",
    width=0.5,
    gap=0.1,
)
plt.title("Relative Band Power (RBP) by Frequency Band and Group")
plt.xlabel("Frequency Band")
plt.ylabel("RBP")
plt.legend()
plt.savefig("../../reports/figures/rbp_by_band_and_group_boxplot", bbox_inches="tight")
plt.show()
