import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


class DatabaseHandler:
    def __init__(self):
        db_path = r"C:\Users\Kishl\OneDrive\Documents\college project semester 1 python\outputs\function_mapping.db"
        self.db_name = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        print(f"Connected to database: {db_path}")

    def load_csv(self, filepath):
        try:
            df = pd.read_csv(filepath)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.columns = ['X', 'Y1(training func)', 'Y2(training func)', 'Y3(training func)', 'Y4(training func)']
            print(f"CSV loaded successfully from {filepath}")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except Exception as e:
            print(f"Unexpected error loading CSV: {e}")
            return None

    def create_table_from_df(self, df, table_name):
        try:
            df.to_sql(table_name, con=self.engine, if_exists="replace", index=False)
            print(f"Table '{table_name}' created and populated.")
        except SQLAlchemyError as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"Unexpected error while creating table: {e}")

    def display_table_preview(self, table_name, limit=5):
        query = text(f"SELECT * FROM {table_name} LIMIT {limit}")
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query)
                rows = result.fetchall()
                print(rows)
                print(f"\n--- First {limit} rows from '{table_name}' ---")
                for row in rows:
                    print(row)
        except SQLAlchemyError as e:
            print(f"SQLAlchemy error: {e}")
        except Exception as e:
            print(f"Unexpected error during query: {e}")


# === Usage Example ===
if __name__ == "__main__":
    db_handler = DatabaseHandler()
    df = db_handler.load_csv(r"C:\Users\Kishl\OneDrive\Documents\college project semester 1 python\data\train.csv")
    if df is not None:
        db_handler.create_table_from_df(df, "training_table")
        db_handler.display_table_preview("training_table")



def load_train_data(filepath):
    db = DatabaseHandler()
    return db.load_csv(filepath)

