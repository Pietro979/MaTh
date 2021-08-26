from sqlalchemy import select, MetaData, Table, insert, Column, String, Sequence, Integer, Float
import matplotlib.pyplot as plt

from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, Float, ForeignKey, and_
from Databases.DataScratching.uploadProceduresNBP import engine_create
import pandas as pd


class PrepareData:
    def __init__(self, database, rate_id):
        for table_name in database.engine.table_names():
            database.dic_table[table_name] = Table(table_name, database.metadata, autoload=True,
                                                        autoload_with=database.engine)
        mapper_stmt = select([database.dic_table['maintable'].columns.value]).where(
            database.dic_table['maintable'].columns.rate_id == rate_id).order_by(
            database.dic_table['maintable'].columns.index.asc())
        print(mapper_stmt)
        mapper_results = database.engine.execute(mapper_stmt).fetchall()
        rate_1_values = [mapper_result[0] for mapper_result in mapper_results]
        # plt.plot(rate_1_values)
        # plt.show()
        df = pd.DataFrame(data=rate_1_values, columns=['Value'])
        df.index += 1
        self.column_indices = {name: i for i, name in enumerate(df.columns)}
        n = len(df)
        train_df = df[0:int(n * 0.7)]
        val_df = df[int(n * 0.7):int(n * 0.9)]
        test_df = df[int(n * 0.9):]

        num_features = df.shape[1]
        # Normalization
        self.train_mean = train_df.mean()
        self.train_std = train_df.std()

        self.train_df = (train_df - self.train_mean) / self.train_std
        self.val_df = (val_df - self.train_mean) / self.train_std
        self.test_df = (test_df - self.train_mean) / self.train_std

        self.df_std = (df - self.train_mean) / self.train_std