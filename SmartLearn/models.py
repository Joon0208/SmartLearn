from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


db = SQLAlchemy()
class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    non_center_eye_count = db.Column(db.Integer)
    elapsed_time_seconds = db.Column(db.Integer)

    def __init__(self, name, non_center_eye_count, elapsed_time_seconds):
        self.name = name
        self.non_center_eye_count = non_center_eye_count
        self.elapsed_time_seconds = elapsed_time_seconds