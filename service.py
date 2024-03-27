import bentoml
import joblib

# Load the trained linear regression model for movie recommendations
model = joblib.load('linear_regression_model.joblib')

# Define the BentoML service
@bentoml.service(resources={"cpu": "1"})
class MovieRecommender:
    @bentoml.api()
    def predict(self, data: dict) -> dict:
        """
        Accepts JSON input in the form of {"userId": int, "movieId": int}
        and returns a JSON output with the predicted rating.
        """
        user_id = data.get("userId")
        movie_id = data.get("movieId")

        if user_id is None or movie_id is None:
            return {"error": "Please provide both userId and movieId"}

        predicted_rating = model.predict([[user_id, movie_id]])[0]
        return {"userId": user_id, "movieId": movie_id, "predictedRating": predicted_rating}
