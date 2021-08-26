from sqlalchemy import select, MetaData, Table, insert, Column, String, Sequence, Integer, Float
import matplotlib.pyplot as plt

from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, Float, ForeignKey, and_
from Databases.DataScratching.uploadProceduresNBP import engine_create
import pandas as pd

Base = declarative_base()


class Maintable(Base):
    __tablename__ = 'maintable'
    index = Column(Integer, primary_key=True)
    date_id = Column(Integer, ForeignKey('dates.date_id'))
    rate_id = Column(Integer, ForeignKey('rates.rate_id'))
    value = Column(Float)
    print("lala")

    def __repr__(self):
        return "<authors(id='{0}', date={1}, value={2})>".format(
            self.id, self.date, self.value)


class Rate(Base):
    __tablename__ = 'rates'
    rate_id = Column(Integer, primary_key=True)
    bank_name = Column(String)
    rate = Column(String)

    def __repr__(self):
        return "<authors(id='{0}', date={1}, value={2})>".format(
            self.id, self.date, self.value)


class Date(Base):
    __tablename__ = 'dates'
    date_id = Column(Integer, primary_key=True)
    date = Column(Date)

    def __repr__(self):
        return "<authors(id='{0}', date={1}, value={2})>".format(
            self.id, self.date, self.value)


class ConnectDatabase:
    def __init__(self):
        self.engine = engine_create()
        self.metadata = MetaData()
        self.dic_table = {}
        self.session = (sessionmaker(bind=self.engine))()
        for table_name in self.engine.table_names():
            self.dic_table[table_name] = Table(table_name, self.metadata, autoload=True, autoload_with=self.engine)

    def choose_data(self, start_date, days, rate):
        # mapper_stmt = select([self.dic_table['maintable'], self.dic_table['dates']]).select_from(
        #     self.dic_table['maintable'].join(self.dic_table['dates'],
        #                                 self.dic_table['maintable'].c.date_id == self.dic_table['dates'].c.date_id)).where(
        #     and_(self.dic_table['maintable'].columns.rate_id == rate,
        #          self.dic_table['dates'].columns.date >= start_date)).order_by(self.dic_table['dates'].columns.date.asc()).limit(
        #     days)
        # session_stmt = q = self.session.query(Maintable, Date)
        # mapper_results = self.engine.execute(mapper_stmt).fetchall()
        #
        # values = [mapper_result[3] for mapper_result in mapper_results]
        # values_presented_df = pd.DataFrame(data=values, columns=['Value'])
        # return values_presented_df
        mapper_stmt = select([self.dic_table['maintable'], self.dic_table['dates']]).select_from(
            self.dic_table['maintable'].join(self.dic_table['dates'],
                                        self.dic_table['maintable'].c.date_id == self.dic_table['dates'].c.date_id)).where(
            and_(self.dic_table['maintable'].columns.rate_id == rate,
                 self.dic_table['dates'].columns.date >= start_date)).order_by(self.dic_table['dates'].columns.date.asc()).limit(
            days)

        print(mapper_stmt)
        session_stmt = q = self.session.query(Maintable, Date)
        mapper_results = self.engine.execute(mapper_stmt).fetchall()

        values = [mapper_result[3] for mapper_result in mapper_results]
        dates = [mapper_result[5].strftime("%m-%d-%Y") for mapper_result in mapper_results]
        d = {'Date': dates, 'Value': values}
        values_presented_df = pd.DataFrame(data=d)
        return values_presented_df




