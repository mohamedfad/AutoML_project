
import pandas as pd
import streamlit as st
from eda import plot_selected_columns
from eda import plot_correlation
from preprocessing import split_features_target,detect_problem_type, split_train_test,build_preprocessor
from model import get_models, compare_models, select_best_model,save_best_model
from tunning import hyperparameter,compare_tunning_models,select_best_model_tunning,save_best_model_tunning


with st.sidebar:
    st.title("AutoML project")

    choice = st.radio(
        "Navigation",
        ["Upload", "EDA", "Preprocessing", "Modelling","tunning"]
    )


if choice == "Upload":
    st.title("Upload Dataset")

    file = st.file_uploader(
        "Upload your dataset",
        type=["csv"]
    )

    if file is not None:
        try:
            df = pd.read_csv(file)
            

            st.session_state.df = df

            df.to_csv("dataset.csv", index=False)

            st.success("Dataset uploaded successfully!")

        except Exception as e:
            st.error(f"Error loading dataset: {e}")

        if st.session_state.df is not None:
            st.dataframe(st.session_state.df)
        
        missing_values = df.isnull().sum()
        missing_columns = missing_values[missing_values > 0]

        if missing_columns.empty:
            st.success("No missing values found!")
        else:
            st.warning("Missing values found!")
            st.dataframe(missing_columns)



elif choice == "EDA":
    st.title("Exploratory Data Analysis")

    if st.session_state.df is not None:
        df = st.session_state.df

        selected_columns = st.multiselect(
            "Select columns to visualize",
            df.columns
        )

        if selected_columns:
            st.title('Plot')
            plot_selected_columns(df, selected_columns)
            st.title('Correlation')
            plot_correlation(df, selected_columns)

        else:
            st.info("Select one or more columns to display their distributions.")

    else:
        st.warning("Please upload a dataset first.")



elif choice == 'Preprocessing':
    df = pd.read_csv('dataset.csv')

    st.title('Preprocessing')

    target_column = st.selectbox(
        'Select Target Column',
        df.columns
    )

       
    suggested_problem_type = detect_problem_type(df, target_column)

    st.session_state['suggested_problem_type'] = suggested_problem_type

    # Save selected target column
    st.session_state['target_column'] = target_column



    if st.button('Process Data'):
        X, y = split_features_target(df, target_column)

        X_train, X_test, y_train, y_test = split_train_test(X, y)

        st.success('Data split successfully!')

        st.write('X_train shape:', X_train.shape)
        st.write('X_test shape:', X_test.shape)
        st.write('y_train shape:', y_train.shape)
        st.write('y_test shape:', y_test.shape)

        st.success('Preprocessing completed successfully!')


        preprocessor = build_preprocessor(X_train)

        X_train_processed = preprocessor.fit_transform(X_train)

        X_test_processed = preprocessor.transform(X_test)
        st.write('Processed X_train shape:', X_train_processed.shape)
        st.write('Processed X_test shape:', X_test_processed.shape)

        st.session_state['X_train_processed'] = X_train_processed
        st.session_state['X_test_processed'] = X_test_processed
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['preprocessor'] = preprocessor


elif choice == 'Modelling':
    st.title('Model Selection')

    # Check if target column was selected
    if 'target_column' not in st.session_state:
        st.warning('Please select Target Column first.')
        st.stop()


    
    suggested_problem_type = st.session_state['suggested_problem_type']

    problem_options = ['Regression', 'Classification']

    
    problem_type_key = f"problem_type_{st.session_state['target_column']}"

    
    if problem_type_key not in st.session_state:
        st.session_state[problem_type_key] = suggested_problem_type

    
    problem_type = st.selectbox(
        'Select Problem Type',
        problem_options,
        key=problem_type_key
    )

    # Save the final user choice
    st.session_state['problem_type'] = problem_type
    
    available_models = get_models(problem_type)

    selected_models = st.multiselect(
        'Select Models to Train',
        options=list(available_models.keys())
    )

    if st.button('Train Models'):
        if not selected_models:
            st.warning('Please select at least one model.')

        elif 'X_train_processed' not in st.session_state:
            st.warning('Please complete Preprocessing first.')

        else:
            with st.spinner('Training models...'):
                results_df, trained_models = compare_models(
                    selected_models=selected_models,
                    available_models=available_models,
                    X_train=st.session_state['X_train_processed'],
                    X_test=st.session_state['X_test_processed'],
                    y_train=st.session_state['y_train'],
                    y_test=st.session_state['y_test'],
                    problem_type=problem_type
                )

                st.session_state['model_results'] = results_df
                st.session_state['trained_models'] = trained_models
                st.session_state['problem_type'] = problem_type

            st.success('Model training completed!')

        
    if 'model_results' in st.session_state:
        st.subheader('Model Comparison')

        results_df = st.session_state['model_results']

        st.dataframe(
            results_df,
            use_container_width=True
        )

        best_model_name, best_model = select_best_model(
            results_df,
            st.session_state['trained_models'],
            st.session_state['problem_type']
        )

        st.session_state['best_model'] = best_model
        st.session_state['best_model_name'] = best_model_name

        st.success(f'Best Model: {best_model_name}')

        problem_type = st.session_state['problem_type']

        if problem_type == 'Regression':
            best_result = results_df.loc[
                results_df['Model'] == best_model_name, 'RMSE'
            ].iloc[0]

            st.metric('Best RMSE', f'{best_result:.4f}')

        else:
            best_result = results_df.loc[
                results_df['Model'] == best_model_name, 'F1-score'
            ].iloc[0]

            st.metric('Best F1-score', f'{best_result:.4f}')

        st.download_button(
        label="Download Best Model",
        data=save_best_model(
            st.session_state['best_model'],
            st.session_state['preprocessor']
        ),
        file_name="best_model.pkl",
        mime="application/octet-stream"
)



elif choice == 'tunning':
    st.title('tunning models')


    problem_type = st.session_state['problem_type']

    available_models = hyperparameter(problem_type)

    selected_models = st.multiselect(
        'Select Tunning Models to Train',
        options=list(available_models.keys())
    )

    if st.button('Train Tunning Models'):
        if not selected_models:
            st.warning('Please select at least one model.')

        elif 'X_train_processed' not in st.session_state:
            st.warning('Please complete Preprocessing first.')

        else:
            with st.spinner('Training  tunning models...'):
                results_df, trained_models = compare_tunning_models(
                    selected_models=selected_models,
                    available_models=available_models,
                    X_train=st.session_state['X_train_processed'],
                    X_test=st.session_state['X_test_processed'],
                    y_train=st.session_state['y_train'],
                    y_test=st.session_state['y_test'],
                    problem_type=problem_type
                )

                st.session_state['tuning_results'] = results_df
                st.session_state['tuned_models'] = trained_models
                st.session_state['tuning_problem_type'] = problem_type

            st.success('Tunning Model training completed!')

        
    if 'tuning_results' in st.session_state:
        st.subheader('Tunning Model Comparison')

        results_df = st.session_state['tuning_results']

        st.dataframe(
            results_df,
            use_container_width=True
        )

        best_model_name, best_model = select_best_model_tunning(
            results_df,
            st.session_state['tuned_models'],
            st.session_state['tuning_problem_type']
        )

        st.session_state['best_tuned_model'] = best_model
        st.session_state['best_tuned_model_name'] = best_model_name

        st.success(f'Best Model: {best_model_name}')

        problem_type = st.session_state['problem_type']

        if problem_type == 'Regression':
            best_result = results_df.loc[
                results_df['Model'] == best_model_name, 'RMSE'
            ].iloc[0]

            st.metric('Best RMSE', f'{best_result:.4f}')

        else:
            best_result = results_df.loc[
                results_df['Model'] == best_model_name, 'F1-score'
            ].iloc[0]

            st.metric('Best F1-score', f'{best_result:.4f}')

        st.download_button(
        label="Download Best Model",
        data=save_best_model_tunning(
            st.session_state['best_tuned_model'],
            st.session_state['preprocessor']
        ),
        file_name="best_model.pkl",
        mime="application/octet-stream"
)