from wtforms import Form, StringField, IntegerField, DateField, RadioField, SelectField, TextAreaField, validators
from wtforms.validators import InputRequired, Email

class CreateUserForm(Form):
    first_name = StringField('First Name', [validators.Length(min=1, max=150), validators.DataRequired()])
    last_name = StringField('Last Name', [validators.Length(min=1, max=150), validators.DataRequired()])
    birthday = DateField('Birthday', [validators.DataRequired()])
    gender = SelectField('Gender', [validators.DataRequired()], choices=[('', 'Select'), ('F', 'Female'), ('M', 'Male')], default='')
    email = StringField('Email', [validators.Length(min=1, max=150), validators.DataRequired()])
    phone_number = StringField('Phone Number',[validators.Length(min=1,max=20)])
    password = StringField('Password', [validators.Length(min=1, max=150), validators.DataRequired()])

class LogIn(Form):
    email = StringField('Email', [validators.Length(min=1, max=150), validators.DataRequired()])
    password = StringField('Password', [validators.Length(min=1, max=150), validators.DataRequired()])



