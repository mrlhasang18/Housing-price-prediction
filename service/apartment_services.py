import sys
import os

# Add the project root directory to Python's module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Rest of your imports
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from domain.domain import ApartmentRequest, ApartmentResponse

class ApartmentService():
    def __init__(self):
        self.path_model = "artifacts/RandomforestApartmentPrice.pkl"
        self.path_encoder = "artifacts/neighbourhood_encoder.pkl"
        self.model = self.load_artifact(self.path_model)
        self.le = self.load_artifact(self.path_encoder)

    def load_artifact(self, path_to_artifact):
        '''Load from a pickle file.'''
        with open(path_to_artifact, 'rb') as f:
            artifact = pickle.load(f)
        return artifact

    def preprocess_input(self, request: ApartmentRequest)->pd.DataFrame:
        data_dict = {
            "rooms": request.rooms,
            "size": request.size,
            "bathrooms": request.bathrooms,
            "neighbourhood": request.neighbourhood,
            "year_built": request.year_built
        } 
        data_df = pd.DataFrame.from_dict([data_dict])

        data_df.neighbourhood = data_df.neighbourhood.str.lower()
        
        # Handle unseen categories
        known_categories = set(self.le.classes_)
        if data_df.neighbourhood[0] not in known_categories:
            print(f"Warning: Unknown neighbourhood '{data_df.neighbourhood[0]}'. Using most frequent category.")
            most_frequent_category = self.le.classes_[0]  # Assuming classes are ordered by frequency
            data_df.neighbourhood = most_frequent_category

        data_df.neighbourhood = self.le.transform(data_df.neighbourhood)
        data_df.neighbourhood = data_df.neighbourhood.astype('category')
        return data_df

    def predict_price(self, request: ApartmentRequest)->ApartmentResponse:
        input_df = self.preprocess_input(request)
        # Predict
        apartment_price = self.model.predict(input_df)[0]
        apartment_price = int(apartment_price)

        response = ApartmentResponse(price=apartment_price)
        return response

if __name__ == "__main__":
    test_request = ApartmentRequest(rooms=2, size=54, bathrooms=1, neighbourhood="Manastur", year_built=1990)

    apt_serv = ApartmentService()
    res = apt_serv.predict_price(request=test_request)
    print(res.price)