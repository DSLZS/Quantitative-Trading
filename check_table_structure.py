from sqlalchemy import create_engine, text

engine = create_engine('mysql+pymysql://root:123456@localhost:3306/quantitative_trading')
with engine.connect() as conn:
    result = conn.execute(text('DESCRIBE stock_fund_flow'))
    for row in result:
        print(row)