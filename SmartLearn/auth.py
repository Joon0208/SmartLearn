from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from models import User
from werkzeug.security import generate_password_hash, check_password_hash
from __init__ import db
from flask_login import login_user, login_required, logout_user, current_user
import os 

# blueprint for flask application
auth = Blueprint('auth', __name__, static_folder='', template_folder='templates/auth_templates')


@auth.route('/login', methods=['GET', 'POST'])
def login():
    # When you access request inside of a route, it will have information about the request that was sent to access this route 
    # It will say the URL, the method
    # We can access the form attribute of our request; has all of the data that was sent as a part of the form
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfullly!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('views.home'))
            else:
                flash('Incorrect password', category='error')
        else:
            flash('Email does not exist', category='error')


    # login_url = url_for('auth.callback', _external=True)
    return render_template('login.html', user=current_user)


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        # Get information from forms 
        email = request.form.get('email')
        name = request.form.get('name')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        user = User.query.filter_by(email=email).first()

        if user:
            flash('Email already exists', category='error')
        elif len(email) < 4:
            flash('Invalid Email', category='error')
        elif len(name) < 2:
            flash('Invalid Name', category='error')
        elif password1 != password2:
            flash('Passwords do not match', category='error')
        elif len(password1) < 7:
            flash('Invalid Email', category='error')
        else:
            # add user to databae
            new_user = User(email=email, name=name, password=generate_password_hash(password1, method='sha256'), role='Student')
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)

            flash("account created", category='success')
            return redirect(url_for('views.home'))

        
    return render_template('signup.html', user=current_user)


@auth.route("/profile")
def profile():

    return render_template('profile.html', user=current_user)
