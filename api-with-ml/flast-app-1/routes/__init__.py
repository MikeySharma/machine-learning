from flask import Blueprint
from .predict import predict_bp

# Create a Blueprint to register routes
def create_routes(app):
    app.register_blueprint(predict_bp)
