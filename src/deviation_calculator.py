import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


class IdealFunctionDatabaseHandler:
    def __init__(self):
        db_path = r"C:\Users\Kishl\OneDrive\Documents\college project semester 1 python\outputs\function_mapping.db"
        self.db_name = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        print(f"Connected to database: {db_path}")

    def load_table_from_db(self, table_name):
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", con=conn)
                print(f"Loaded table '{table_name}' successfully.")
                return df
        except Exception as e:
            print(f"Error loading table '{table_name}': {e}")
            return None

    def preview_table(self, table_name, limit=5):
        query = text(f"SELECT * FROM {table_name} LIMIT {limit}")
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query)
                rows = result.fetchall()
                print(f"\n--- Preview: {table_name} ---")
                for row in rows:
                    print(row)
        except SQLAlchemyError as e:
            print(f"SQLAlchemy error: {e}")
        except Exception as e:
            print(f"Unexpected error during query: {e}")

    def find_best_matching_ideals(self, train_df, ideal_df):
        if not train_df['X'].equals(ideal_df['X']):
            print("X-values do not match between training and ideal datasets.")
            return

        matches = {}
        for train_col in ['Y1(training func)', 'Y2(training func)', 'Y3(training func)', 'Y4(training func)']:
            best_col = None
            min_deviation = float('inf')
            for ideal_col in ideal_df.columns[1:]:  # Skip X
                deviation = ((train_df[train_col] - ideal_df[ideal_col]) ** 2).sum()
                if deviation < min_deviation:
                    min_deviation = deviation
                    best_col = ideal_col
            matches[train_col] = (best_col, min_deviation)

        print("\n--- Best Matching Ideal Functions ---")
        for train_func, (ideal_func, deviation) in matches.items():
            print(f"{train_func} -> {ideal_func} with total deviation {deviation:.4f}")


# === Usage ===
if __name__ == "__main__":
    handler = IdealFunctionDatabaseHandler()

    ideal_df = handler.load_table_from_db("ideal_training_table")
    train_df = handler.load_table_from_db("training_table")

    if ideal_df is not None:
        handler.preview_table("ideal_training_table")

    if train_df is not None:
        handler.preview_table("training_table")

    if train_df is not None and ideal_df is not None:
        handler.find_best_matching_ideals(train_df, ideal_df)




def calculate_ideal_mapping(train_data, ideal_data):
    calculator = IdealFunctionDatabaseHandler()
    return calculator.find_best_mappings()
