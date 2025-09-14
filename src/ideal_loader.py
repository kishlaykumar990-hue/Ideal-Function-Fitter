import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


class IdealFunctionDatabaseHandler:
    def __init__(self):
        db_path = r"C:\Users\Kishl\OneDrive\Documents\college project semester 1 python\outputs\function_mapping.db"
        self.db_name = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        print(f"Connected to database: {db_path}")

    def load_csv(self, filepath):
        try:
            df = pd.read_csv(filepath)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            column_names = ['X'] + [f"Y{i}" for i in range(1, 51)]
            df.columns = column_names
            print(f"CSV loaded and columns renamed for ideal functions.")
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None

    def create_table_from_df(self, df, table_name):
        try:
            df.to_sql(table_name, con=self.engine, if_exists="replace", index=False)
            print(f"Table '{table_name}' created successfully.")
        except SQLAlchemyError as e:
            print(f"SQLAlchemy Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

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


# === Usage ===
if __name__ == "__main__":
    handler = IdealFunctionDatabaseHandler() 
    df = handler.load_csv(r"C:\Users\Kishl\OneDrive\Documents\college project semester 1 python\data\ideal.csv")
    if df is not None:
        handler.create_table_from_df(df, "ideal_training_table")
        handler.preview_table("ideal_training_table")



def load_ideal_data(filepath):
    db = IdealFunctionDatabaseHandler()
    return db.load_csv(filepath)
