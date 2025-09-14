import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10, Dark2

class DataHandler:
    """Base class for handling data loading from database and CSV files."""
    
    def __init__(self, db_path):
        """Initialize database connection."""
        self.engine = create_engine(f"sqlite:///{db_path}")
        print(f"Connected to database: {db_path}")

    def load_table_from_db(self, table_name):
        """Load a table from the SQLite database."""
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", con=conn)
                print(f"Loaded table '{table_name}' successfully. Columns: {list(df.columns)}")
                return df
        except SQLAlchemyError as e:
            print(f"SQLAlchemy error loading table '{table_name}': {e}")
            return None
        except Exception as e:
            print(f"Unexpected error loading table '{table_name}': {e}")
            return None

    def load_test_csv(self, filepath):
        """Load test CSV file and remove unnamed columns."""
        try:
            df = pd.read_csv(filepath)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            print(f"Test CSV loaded successfully from {filepath}. Columns: {list(df.columns)}")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except Exception as e:
            print(f"Unexpected error loading test CSV: {e}")
            return None

    def load_train_csv(self, filepath):
        """Load train CSV file and remove unnamed columns."""
        try:
            df = pd.read_csv(filepath)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            print(f"Train CSV loaded successfully from {filepath}. Columns: {list(df.columns)}")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except Exception as e:
            print(f"Unexpected error loading train CSV: {e}")
            return None

class DeviationRegressionHandler(DataHandler):
    # Class constants
    DB_PATH = r"C:\Users\Kishl\OneDrive\Documents\college project semester 1 python\outputs\function_mapping.db"
    TEST_CSV_PATH = r"C:\Users\Kishl\OneDrive\Documents\college project semester 1 python\data\test.csv"
    TRAIN_CSV_PATH = r"C:\Users\Kishl\OneDrive\Documents\college project semester 1 python\data\train.csv"
    IDEAL_TABLE = "ideal_training_table"
    TEST_TABLE = "test_deviations"
    TRAIN_TABLE = "train_deviations"
    IDEAL_COLUMNS = ['Y42', 'Y41', 'Y11', 'Y48']
    TRAIN_COLUMNS = ['y1', 'y2', 'y3', 'y4']
    SQRT_2 = 1.4142135623730951  # Updated for precise Max Deviation calculations

    def __init__(self):
        """Initialize DeviationRegressionHandler with database connection."""
        super().__init__(self.DB_PATH)

    def interpolate_ideal_values(self, test_df, ideal_df, ideal_columns):
        """Interpolate ideal values for test.csv."""
        # Verify that all ideal_columns exist in ideal_df
        missing_cols = [col for col in ideal_columns if col not in ideal_df.columns]
        if missing_cols:
            print(f"Error: Columns {missing_cols} not found in ideal_training_table.")
            return None

        # Initialize result DataFrame
        result_df = test_df.copy()
        for col in ideal_columns:
            result_df[f"{col}_interp"] = np.nan

        # Ensure ideal_df is sorted by X
        ideal_df = ideal_df.sort_values(by='X')

        for i, row in test_df.iterrows():
            x_test = row['x']
            y_test = row['y']
            # Debug output for x = 17.5
            if abs(x_test - 17.5) < 1e-6:
                print(f"\nDebug: For x = {x_test}, y = {y_test}, raw values from ideal_training_table:")
                try:
                    raw_row = ideal_df.loc[ideal_df['X'] == x_test]
                    if not raw_row.empty:
                        for col in ideal_columns:
                            print(f"{col}: {raw_row[col].iloc[0]}")
                    else:
                        print(f"No exact match for x = {x_test}")
                except Exception as e:
                    print(f"Error accessing raw values: {e}")

            # Check if x_test is within the range of ideal_df
            if x_test < ideal_df['X'].min() or x_test > ideal_df['X'].max():
                print(f"Warning: x={x_test} is outside ideal data range. Skipping interpolation.")
                continue

            # Find the two closest x values in ideal_df
            lower = ideal_df[ideal_df['X'] <= x_test]['X'].max()
            upper = ideal_df[ideal_df['X'] >= x_test]['X'].min()

            if lower == upper:  # Exact match
                for col in ideal_columns:
                    raw_value = ideal_df.loc[ideal_df['X'] == x_test, col].iloc[0]
                    result_df.at[i, f"{col}_interp"] = raw_value
            else:
                # Linear interpolation
                lower_row = ideal_df[ideal_df['X'] == lower].iloc[0]
                upper_row = ideal_df[ideal_df['X'] == upper].iloc[0]
                x0, x1 = lower_row['X'], upper_row['X']
                for col in ideal_columns:
                    y0, y1 = lower_row[col], upper_row[col]
                    y_interp = y0 + (y1 - y0) * (x_test - x0) / (x1 - x0)
                    result_df.at[i, f"{col}_interp"] = y_interp

            # Debug deviations for x = 17.5
            if abs(x_test - 17.5) < 1e-6:
                for col in ideal_columns:
                    print(f"dev_{col} = abs(y={y_test} - {col}_interp={result_df.at[i, f'{col}_interp']}) = {np.abs(y_test - result_df.at[i, f'{col}_interp'])}")

        return result_df

    def compute_train_deviations(self, train_df, ideal_df, train_cols, ideal_cols):
        """Compute deviations for train.csv against ideal functions."""
        # Verify that all required columns exist
        missing_train_cols = [col for col in train_cols if col not in train_df.columns]
        missing_ideal_cols = [col for col in ideal_cols if col not in ideal_df.columns]
        if missing_train_cols or missing_ideal_cols:
            print(f"Error: Missing columns in train.csv: {missing_train_cols}, ideal_training_table: {missing_ideal_cols}")
            return None

        # Initialize result DataFrame
        result_df = train_df.copy()
        for ideal_col in ideal_cols:
            result_df[ideal_col] = np.nan
        deviation_cols = [f"dev_{train_col}_{ideal_col}" for train_col, ideal_col in zip(train_cols, ideal_cols)]
        for dev_col in deviation_cols:
            result_df[dev_col] = np.nan

        # Ensure ideal_df is sorted by X
        ideal_df = ideal_df.sort_values(by='X')

        for i, row in train_df.iterrows():
            x_train = row['x']
            # Debug output for x = 17.5
            if abs(x_train - 17.5) < 1e-6:
                print(f"\nDebug: For x = {x_train} in train.csv, raw values:")
                print(f"train.csv: y1={row['y1']}, y2={row['y2']}, y3={row['y3']}, y4={row['y4']}")
                try:
                    raw_row = ideal_df.loc[ideal_df['X'] == x_train]
                    if not raw_row.empty:
                        for col in ideal_cols:
                            print(f"ideal_training_table {col}: {raw_row[col].iloc[0]}")
                    else:
                        print(f"No exact match for x = {x_train}")
                except Exception as e:
                    print(f"Error accessing raw values: {e}")

            # Check if x_train is within the range of ideal_df
            if x_train < ideal_df['X'].min() or x_train > ideal_df['X'].max():
                print(f"Warning: x={x_train} is outside ideal data range. Skipping interpolation.")
                continue

            # Find the two closest x values in ideal_df
            lower = ideal_df[ideal_df['X'] <= x_train]['X'].max()
            upper = ideal_df[ideal_df['X'] >= x_train]['X'].min()

            if lower == upper:  # Exact match
                for ideal_col in ideal_cols:
                    result_df.at[i, ideal_col] = ideal_df.loc[ideal_df['X'] == x_train, ideal_col].iloc[0]
            else:
                # Linear interpolation
                lower_row = ideal_df[ideal_df['X'] == lower].iloc[0]
                upper_row = ideal_df[ideal_df['X'] == upper].iloc[0]
                x0, x1 = lower_row['X'], upper_row['X']
                for ideal_col in ideal_cols:
                    y0, y1 = lower_row[ideal_col], upper_row[ideal_col]
                    y_interp = y0 + (y1 - y0) * (x_train - x0) / (x1 - x0)
                    result_df.at[i, ideal_col] = y_interp

            # Calculate deviations
            for train_col, ideal_col, dev_col in zip(train_cols, ideal_cols, deviation_cols):
                result_df.at[i, dev_col] = np.abs(row[train_col] - result_df.at[i, ideal_col])
                # Debug deviation for x = 17.5
                if abs(x_train - 17.5) < 1e-6:
                    print(f"{dev_col} = abs({train_col}={row[train_col]} - {ideal_col}={result_df.at[i, ideal_col]}) = {result_df.at[i, dev_col]}")

        return result_df

    def prepare_output(self, test_df, ideal_columns):
        """Prepare test_deviations with deviations and limit checks."""
        result_df = test_df.copy()
        output_columns = ['x', 'y'] + [f"{col}_interp" for col in ideal_columns]

        # Calculate deviations: abs(y - YXX_interp)
        for col in ideal_columns:
            result_df[f"dev_{col}"] = np.abs(result_df['y'] - result_df[f"{col}_interp"])
        output_columns += [f"dev_{col}" for col in ideal_columns]

        # Define max deviation thresholds from train_deviations
        max_dev_thresholds = {
            'dev_Y42': 0.7014046719180178,
            'dev_Y41': 0.7038583324480879,
            'dev_Y11': 0.7056020577700327,
            'dev_Y48': 0.7067413340734435
        }

        # Add limit columns
        for col in ideal_columns:
            dev_col = f"dev_{col}"
            limit_col = f"limit_{col}"
            result_df[limit_col] = result_df[dev_col].apply(
                lambda x: "crosses limit" if x > max_dev_thresholds[dev_col] else "within limit"
            )
            output_columns.append(limit_col)

        # Rename columns as requested
        result_df = result_df.rename(columns={
            'x': 'X',
            'y': 'Y',
            'Y42_interp': 'Y42',
            'Y41_interp': 'Y41',
            'Y11_interp': 'Y11',
            'Y48_interp': 'Y48'
        })

        # Update output_columns to reflect renamed columns
        output_columns = ['X', 'Y'] + [col for col in ideal_columns] + [f"dev_{col}" for col in ideal_columns] + [f"limit_{col}" for col in ideal_columns]
        result_df = result_df[output_columns]
        return result_df

    def create_final_table(self, test_result_df, ideal_columns):
        """Create final table with X, Y, Delta Y (test func), and No. of ideal func."""
        final_df = test_result_df[['X', 'Y']].copy()
        final_df['Delta Y (test func)'] = np.nan
        final_df['No. of ideal func'] = ''

        for i, row in test_result_df.iterrows():
            # Collect deviations and corresponding ideal functions where limit is "within limit"
            within_limit_devs = {}
            for col in ideal_columns:
                limit_col = f"limit_{col}"
                dev_col = f"dev_{col}"
                if row[limit_col] == "within limit":
                    within_limit_devs[col] = row[dev_col]

            # If there are deviations within limit, select the smallest one
            if within_limit_devs:
                min_dev_col = min(within_limit_devs, key=within_limit_devs.get)
                final_df.at[i, 'Delta Y (test func)'] = within_limit_devs[min_dev_col]
                final_df.at[i, 'No. of ideal func'] = min_dev_col

        # Remove rows where 'Delta Y (test func)' is NaN or 'No. of ideal func' is empty
        final_df = final_df.dropna(subset=['Delta Y (test func)', 'No. of ideal func'])
        final_df = final_df[final_df['No. of ideal func'] != '']

        return final_df

    def visualize_final_table(self, final_df, ideal_df, ideal_columns):
        """
        Create a comprehensive Bokeh plot with:
        - Scatter plot of assigned test data points.
        - Line graphs of the corresponding ideal functions.
        - Tooltips for detailed data inspection.
        """
        output_file("comprehensive_visualization.html")

        # Create a ColumnDataSource for the final table
        final_source = ColumnDataSource(final_df)
        
        # Define color mapping for 'No. of ideal func' and ideal functions
        ideal_palette = Dark2[len(ideal_columns)]
        color_map = dict(zip(ideal_columns, ideal_palette))

        # Map colors and sizes to the data
        final_df['color'] = final_df['No. of ideal func'].map(color_map)
        final_source.data['color'] = final_df['color']
        
        # Scale point sizes based on Delta Y (test func)
        final_df['size'] = final_df['Delta Y (test func)'].apply(lambda x: 5 + 20 * x)
        final_source.data['size'] = final_df['size']
        
        # Create the figure
        p = figure(title="Test Data Assigned to Ideal Functions",
                    x_axis_label="X",
                    y_axis_label="Y",
                    tools="pan,box_zoom,reset,save",
                    width=1000, height=700)
        
        # Add the ideal function lines
        ideal_df = ideal_df.sort_values(by='X')
        for i, col in enumerate(ideal_columns):
            ideal_source = ColumnDataSource(ideal_df)
            p.line(x='X', y=col, source=ideal_source, legend_label=f"Ideal {col}", 
                   color=color_map[col], line_width=2, line_dash="dashed", alpha=0.7)
        
        # Add the scatter plot for the test data points
        scatter_plot = p.scatter(x='X', y='Y', size='size', color='color', source=final_source,
                                 legend_field='No. of ideal func', alpha=0.8)
        
        # Add tooltips for scatter plot
        hover = HoverTool(renderers=[scatter_plot])
        hover.tooltips = [
            ("X", "@X"),
            ("Y", "@Y"),
            ("Delta Y", "@{Delta Y (test func)}{0.0000}"),
            ("Assigned Ideal Function", "@{No. of ideal func}")
        ]
        p.add_tools(hover)
        
        # Customize legend
        p.legend.title = "Data & Assignments"
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        
        # Save the plot
        save(p)
        print("Bokeh visualization saved to comprehensive_visualization.html")

    def store_results_to_db(self, df, table_name):
        """Store DataFrame to SQLite database."""
        try:
            df.to_sql(table_name, con=self.engine, if_exists="replace", index=False)
            print(f"Table '{table_name}' created and populated with results.")
        except SQLAlchemyError as e:
            print(f"SQLAlchemy error storing table '{table_name}': {e}")
        except Exception as e:
            print(f"Unexpected error storing table '{table_name}': {e}")

    def preview_results(self, df, table_name):
        """Preview DataFrame and save to CSV with deviation summaries."""
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        # Format the 'X' column as strings with one decimal place
        if 'X' in df_copy.columns:
            df_copy['X'] = df_copy['X'].map(lambda x: f"{x:.1f}")

        print(f"\n--- Preview: {table_name} (All rows) ---")
        print(df_copy)
        # Save to CSV for full inspection
        csv_path = f"{table_name.lower().replace(' ', '_')}_preview.csv"
        # Calculate largest and max deviations for train_deviations
        if table_name == "Train Deviations":
            deviation_cols = ['dev_y1_Y42', 'dev_y2_Y41', 'dev_y3_Y11', 'dev_y4_Y48']
            largest_devs = {}
            max_devs = {}
            for col in deviation_cols:
                if col in df_copy.columns:
                    largest_devs[f"Largest {col}"] = df_copy[col].max()
                    max_devs[f"Max Deviation {col}"] = largest_devs[f"Largest {col}"] * self.SQRT_2
                    print(f"Largest {col}: {largest_devs[f'Largest {col}']}")
                    print(f"Max Deviation {col}: {max_devs[f'Max Deviation {col}']}")
            # Save DataFrame to CSV, preserving the formatted 'X' column
            df_copy.to_csv(csv_path, index=False)
            # Append largest and max deviations to CSV
            with open(csv_path, 'a') as f:
                f.write("\nLargest Deviations\n")
                for col, value in largest_devs.items():
                    f.write(f"{col},{value:.6f}\n")
                f.write("\nMax Deviations\n")
                for col, value in max_devs.items():
                    f.write(f"{col},{value:.6f}\n")
        else:
            # Save DataFrame to CSV, preserving the formatted 'X' column
            df_copy.to_csv(csv_path, index=False)
        print(f"Full table saved to {csv_path} for inspection.")

# === Usage ===
if __name__ == "__main__":
    handler = DeviationRegressionHandler()
    
    # Load ideal table from database
    ideal_df = handler.load_table_from_db("ideal_training_table")
    train_df = handler.load_train_csv(r"C:\Users\Kishl\OneDrive\Documents\college project semester 1 python\data\train.csv")
    test_df = handler.load_test_csv(r"C:\Users\Kishl\OneDrive\Documents\college project semester 1 python\data\test.csv")
    
    if ideal_df is None or train_df is None or test_df is None:
        print("Error: Required data could not be loaded. Exiting.")
    else:
        # Select the specified ideal functions for test.csv
        ideal_columns = ['Y42', 'Y41', 'Y11', 'Y48']
        # Columns for train.csv deviations
        train_columns = ['y1', 'y2', 'y3', 'y4']
        
        # Check if columns exist
        missing_ideal_cols = [col for col in ideal_columns if col not in ideal_df.columns]
        missing_train_cols = [col for col in train_columns if col not in train_df.columns]
        if missing_ideal_cols or missing_train_cols:
            print(f"Error: Missing columns in ideal_training_table: {missing_ideal_cols}, train.csv: {missing_train_cols}")
        else:
            # Process train.csv first
            train_result_df = handler.compute_train_deviations(train_df, ideal_df, train_columns, ideal_columns)
            if train_result_df is not None:
                # Store and preview train results
                handler.store_results_to_db(train_result_df, "train_deviations")
                handler.preview_results(train_result_df, "Train Deviations")

            # Process test.csv
            interpolated_df = handler.interpolate_ideal_values(test_df, ideal_df, ideal_columns)
            if interpolated_df is not None:
                # Prepare output for test.csv
                test_result_df = handler.prepare_output(interpolated_df, ideal_columns)
                # Store and preview test results
                handler.store_results_to_db(test_result_df, "test_deviations")
                handler.preview_results(test_result_df, "Test Deviations")
                
                # Create and store final table
                final_df = handler.create_final_table(test_result_df, ideal_columns)
                handler.store_results_to_db(final_df, "final_table")
                handler.preview_results(final_df, "Final Table")
                
                # Visualize final table
                handler.visualize_final_table(final_df, ideal_df, ideal_columns)




def process_test_data(test_data, mappings):
    handler = DeviationRegressionHandler(test_data, mappings)
    return handler.apply_mappings()


