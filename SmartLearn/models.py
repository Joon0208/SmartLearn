from flask_login import UserMixin
from sqlalchemy.sql import func
from sqlalchemy.types import LargeBinary
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


db = SQLAlchemy()




# class Videos(db.Model):
#     id = db.Column(db.Integer, primary_key = True)
#     video = db.Column(db.String(500))

#     # Foreign key is a column that references a column of another database(primary key)
#     #                                            user is from class User 
#     # 1 to many relationship 
#     # Stored a foreign key on the child object referencing the Parent object.1  Parents Object that has many children objects
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'))




# class Question(db.Model):
#     id = db.Column(db.Integer, primary_key = True)
#     question_text = db.Column(db.String(500))
#     options = db.relationship('Option', backref='question', lazy=True)
#     reports = db.relationship('Report', backref='question', lazy=True)

# class Option(db.Model):
#     id = db.Column(db.Integer, primary_key = True)
#     option_text = db.Column(db.String(200), nullable=False)
#     is_correct = db.Column(db.Boolean, default=False)
#     question_id = db.Column(db.Integer, db.ForeignKey('question.id'), nullable=False)

# class Report(db.Model):
#     id = db.Column(db.Integer, primary_key = True)
#     question_id = db.Column(db.Integer, db.ForeignKey('question.id'), nullable=False)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     # timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     time_taken = db.Column(db.Float, nullable=True)
#     is_correct = db.Column(db.Boolean)
    


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    name = db.Column(db.String(150)) 
    date_joined = db.Column(db.Date, default=datetime.utcnow)
    role = db.Column(db.String(20))
    # a foreign key is a column in your database that always references a column of amother database 

    # We want users to be able to find all of their videos
    # Everytime we create a video, add into the users videos relationship the video id 
    # videos = db.relationship('Videos') 
    # questions = db.relationship('Question', backref='user', lazy=True)
    reports = db.relationship('Report', backref='user', lazy=True)
