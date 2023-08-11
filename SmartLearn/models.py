from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class ImageModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(255), nullable=False)
    image_data = db.Column(db.LargeBinary, nullable=False)

    def __init__(self, user_name, image_data):
        self.user_name = user_name
        self.image_data = image_data