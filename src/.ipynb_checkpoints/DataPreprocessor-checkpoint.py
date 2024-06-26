import pandas as pd
from category_encoders import HashingEncoder
import pandas as pd

class DataPreprocessor():
    def __init__(self, df):
        self.df = df
        pass

    def drop_and_one_hot_encode(self):

        columns_to_drop = ['message', 'severity', 'facility', 'time' ,'severity_numbers']
        columns_to_encode =['host', 'ident', 'pid', 'facility_numbers', 'user', 'user_cron',\
                            'sender', 'receiver', 'alphanum_code', 'status', 'ip', 'user_1',\
                            'user_2', 'bet_par', 'session']

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


    def drop_and_hash_encode(self):

        columns_to_drop = ['message', 'severity', 'facility', 'time' ,'severity_numbers']
        columns_to_encode =['host', 'ident', 'pid', 'facility_numbers', 'user', 'user_cron',\
                            'sender', 'receiver', 'alphanum_code', 'status', 'ip', 'user_1',\
                            'user_2', 'bet_par', 'session'] #, 'log key', 'log key spell']

        #['host', 'ident', 'pid', 'message', 'severity', 'facility', 'time','severity_numbers', 'facility_numbers', 'severity_scores', 'timedelta','log key', 'log key spell']

        # One-Hot encode the log keys also !!!

        columns_to_keep = [col for col in self.df.columns if col not in columns_to_drop]
        # Create a new DataFrame without the columns to drop
        self.df = self.df[columns_to_keep].copy()

        columns_to_encode = [col for col in self.df.columns if col in columns_to_encode]

        try:
            encoder = HashingEncoder(n_components=20)
            encoded_data = encoder.fit_transform(self.df[columns_to_encode])
            self.df = pd.concat([self.df, encoded_data], axis=1)
            self.df = self.df.drop(columns = columns_to_encode, axis=1)
        except ValueError as e:
            print(f"Error encoding column {col}: {e}")

        return self.df