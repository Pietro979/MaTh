from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import Column, Integer, String, Date, Float, ForeignKey
import uploadProceduresNBP

Base = declarative_base()

class Maintable(Base):
    __tablename__ = 'maintable'
    index = Column(Integer, primary_key=True)
    date_id = Column(Integer, ForeignKey('dates.date_id'))
    rate_id = Column(Integer, ForeignKey('rates.rate_id'))
    value = Column(Float)


class Rate(Base):
    __tablename__ = 'rates'
    rate_id = Column(Integer, primary_key=True)
    bank_name = Column(String)
    rate = Column(String)


class Dates(Base):
    __tablename__ = 'dates'
    date_id = Column(Integer, primary_key=True)
    date = Column(Date)


engine = uploadProceduresNBP.engine_create()
Base.metadata.create_all(engine)

# Usuwanie danej tabeli

