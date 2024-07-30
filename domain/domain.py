# Pydantic is a Python library for data validation and settings management using Python-type annotations.
#It ensures that the data you work with matches your specified data types, simplifying error handling and data parsing in Python applications.
from pydantic import BaseModel

# for API request
class ApartmentRequest(BaseModel):
    rooms: int
    size: int
    bathrooms: int
    neighbourhood: str
    year_built: int

# for API response: since our response in predicted price
class ApartmentResponse(BaseModel):
    price: int