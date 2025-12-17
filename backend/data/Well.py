
from sqlalchemy import Column, Integer, String, Float

from backend.data.Base import Base


class Well(Base):
    __tablename__ = "well";
    id = Column(Integer, primary_key=True, autoincrement=True);
    name = Column(String);
    depth = Column(Float)
    value = Column(Float)