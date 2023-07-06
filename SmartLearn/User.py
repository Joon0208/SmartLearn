
class User:
    count_id = 0

    def __init__(self, first_name, last_name, birthday, gender, email, phone_number, password):
        User.count_id += 1

        #
        # if len(users_list) == 0:
        #     next_id = 0
        # else:
        #

        self.__user_id = User.count_id
        self.__first_name = first_name
        self.__last_name = last_name
        self.__birthday = birthday
        self.__gender = gender
        self.__email = email
        self.__phone_number = phone_number
        self.__password = password

    def get_user_id(self):
        return self.__user_id

    def get_first_name(self):
        return self.__first_name

    def get_last_name(self):
        return self.__last_name

    def get_birthday(self):
        return self.__birthday

    def get_gender(self):
        return self.__gender

    def get_email(self):
        return self.__email

    def get_phone_number(self):
        return self.__phone_number

    def get_password(self):
        return self.__password

    def set_user_id(self, user_id):
        self.__user_id = user_id

    def set_first_name(self, first_name):
        self.__first_name = first_name

    def set_last_name(self, last_name):
        self.__last_name = last_name

    def set_birthday(self, birthday):
        self.__birthday = birthday

    def set_gender(self, gender):
        self.__gender = gender

    def set_email(self, email):
        self.__email = email

    def set_phone_number(self, phone_number):
        self.__phone_number = phone_number

    def set_password(self, password):
        self.__password = password



