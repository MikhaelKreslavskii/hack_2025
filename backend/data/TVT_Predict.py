
from sqlalchemy import Column, Integer, String, Float

from backend.data.Base import Base


class TVT_Predict(Base):
    __tablename__ = "tvt_predict";
    name = Column(String,primary_key=True);
    x = Column(Float)
    y = Column(Float)
    h_kol = Column(Float)