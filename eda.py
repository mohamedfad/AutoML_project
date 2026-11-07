
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

def plot_selected_columns(df, selected_columns):

    for column in selected_columns:

        if pd.api.types.is_numeric_dtype(df[column]):
            fig, ax = plt.subplots()
            ax.hist(df[column].dropna(), bins=20)
            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")

        else:
            fig, ax = plt.subplots()
            df[column].value_counts().head(20).plot(
                kind="bar", ax=ax
            )
            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)

        st.pyplot(fig)
        plt.close(fig)


def plot_correlation(df, selected_columns):
    numeric_df = df[selected_columns].select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        st.warning("Select at least two numeric columns.")
        return  

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax)

    st.pyplot(fig)
    plt.close(fig)