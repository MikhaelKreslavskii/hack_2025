
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass
class Well(Base):
    __tablename__ = "well";
    id = Column(Integer, primary_key=True);
    name = Column(String);
    depth = Column(Float)
    value = Column(Float)