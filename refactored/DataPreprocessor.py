import pandas as pd

class DataPreprocessor():
    def __init__(self, df):
        self.df = df
        pass

    def drop_and_one_hot_encode(self):

        columns_to_drop = ['message', 'severity', 'facility', 'time' ,'severity_numbers']
        columns_to_encode =['host', 'ident', 'pid', 'facility_numbers', 'user', 'user_cron',\
                            'sender', 'receiver', 'alphanum_code', 'status', 'ip', 'user_1',\
                            'user_2']

        # One-Hot encode the log keys also !!!

        columns_to_keep = [col for col in self.df.columns if col not in columns_to_drop]
        # Create a new DataFrame without the columns to drop
        self.df = self.df[columns_to_keep].copy()

        for col in columns_to_encode:
            if col in self.df.columns:
                try:
                    self.df = pd.get_dummies(self.df, columns=[col])
                    #print(f'encoded {col} from df {i}')
                except ValueError as e:
                    print(f"Error encoding column {col}: {e}")

        return self.df