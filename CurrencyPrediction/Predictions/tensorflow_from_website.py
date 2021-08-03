import pandas as pd
from Databases.DataScratching.uploadProceduresNBP import engine_create

from sqlalchemy import select, MetaData, Table, insert, Column, String, Sequence, Integer, Float
import matplotlib.pyplot as plt

engine = engine_create()
metadata = MetaData()

dic_table = {}

for table_name in engine.table_names():
    dic_table[table_name] = Table(table_name,metadata, autoload = True, autoload_with = engine)

# print(repr(dic_table['maintable']))
mapper_stmt = select([dic_table['maintable']])
print('Mapper select: ')
print(mapper_stmt)

mapper_results = engine.execute(mapper_stmt).fetchall()
# print(mapper_results)

mapper_stmt = select([dic_table['maintable'].columns.value]).where(dic_table['maintable'].columns.rate_id == 1).order_by(dic_table['maintable'].columns.index.desc())
mapper_results = engine.execute(mapper_stmt).fetchall()
rate_1_values = [mapper_result[0] for mapper_result in mapper_results ]
plt.plot(rate_1_values)
plt.show()

rate_1_values_df = pd.DataFrame(data = rate_1_values, columns = ['Date'])
rate_1_values_df.index += 1

print(rate_1_values_df)