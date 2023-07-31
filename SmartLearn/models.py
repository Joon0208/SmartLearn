from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


db = SQLAlchemy()
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    name = db.Column(db.String(150), nullable=False)
    date_joined = db.Column(db.Date, default=datetime.utcnow, nullable=False)
    role = db.Column(db.String(20), nullable=False)
    exam_score = db.Column(db.Integer, default=0, nullable=False)
    cheating_percentage = db.Column(db.Float, default=0.0, nullable=False)

    def __repr__(self):
        return f"<User {self.email}>"
