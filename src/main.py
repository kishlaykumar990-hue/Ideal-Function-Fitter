# src/main.py

import os
import pandas as pd
from src.test_handler import DeviationRegressionHandler


def main():
    print("Starting project pipeline...")

    # Step 1: Initialize handler
    handler = DeviationRegressionHandler()

    # Step 2: Load datasets
    train_df = handler.load_train_csv(handler.TRAIN_CSV_PATH)
    print("Training dataset loaded")

    ideal_df = handler.load_table_from_db(handler.IDEAL_TABLE)
    print("Ideal dataset loaded")

    test_df = handler.load_test_csv(handler.TEST_CSV_PATH)
    print("Test dataset loaded")

    if ideal_df is None or train_df is None or test_df is None:
        print("Error: Required datasets could not be loaded. Exiting.")
        return

    # Step 3: Define columns
    ideal_columns = handler.IDEAL_COLUMNS
    train_columns = handler.TRAIN_COLUMNS

    # Step 4: Process train.csv
    train_result_df = handler.compute_train_deviations(train_df, ideal_df, train_columns, ideal_columns)
    if train_result_df is not None:
        handler.store_results_to_db(train_result_df, handler.TRAIN_TABLE)
        handler.preview_results(train_result_df, "Train Deviations")

    # Step 5: Process test.csv
    interpolated_df = handler.interpolate_ideal_values(test_df, ideal_df, ideal_columns)
    if interpolated_df is not None:
        test_result_df = handler.prepare_output(interpolated_df, ideal_columns)
        handler.store_results_to_db(test_result_df, handler.TEST_TABLE)
        handler.preview_results(test_result_df, "Test Deviations")

        # Step 6: Create final mapping
        final_df = handler.create_final_table(test_result_df, ideal_columns)
        handler.store_results_to_db(final_df, "final_table")
        handler.preview_results(final_df, "Final Table")

        # Step 7: Visualization
        handler.visualize_final_table(final_df, ideal_df, ideal_columns)

    print("Project pipeline finished successfully!")


if __name__ == "__main__":
    main()
