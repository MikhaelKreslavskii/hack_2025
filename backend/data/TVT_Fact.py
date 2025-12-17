
from sqlalchemy import Column, Integer, String, Float

from backend.data.Base import Base


class TVT_Fact(Base):
    __tablename__ = "tvt_fact";
    name = Column(String,primary_key=True);
    x = Column(Float)
    y = Column(Float)
    h_kol = Column(Float)