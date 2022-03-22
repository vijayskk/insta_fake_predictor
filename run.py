import joblib

model = joblib.load('model.joblib')

def ask_yes_or_no(question) -> bool:
    """Ask a yes or no question."""
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return ask_yes_or_no("Uhhhh... please enter ")

def ask_a_integer(question):
    try:
        return int(input(question))
    except ValueError:
        print("Sorry, I didn't understand that.")

propic = ask_yes_or_no("Does he have a profile picture?")
unamecharectercount = ask_a_integer("How many characters does his username have?")
unamenumericalcount = ask_a_integer("How many numerical characters does his username have?")
fullnamewords = ask_a_integer("How many words does his full name have?")
fullnamecharectercount = ask_a_integer("How many characters does his fullname have?")
fullnamenumericalcount = ask_a_integer("How many numerical characters does his fullname have?")
unameequalsfullname = ask_yes_or_no("Does his username and full name literally the same?")
biolength = ask_a_integer("How many characters does his bio have?")
exturl = ask_yes_or_no("Does he have an external url in bio?")
private = ask_yes_or_no("Is his profile private?")
postcount = ask_a_integer("How many posts does he have?")
followers = ask_a_integer("How many followers does he have?")
following = ask_a_integer("How many people does he follow?")


pred = model.predict([[
    propic,
    (unamenumericalcount / unamecharectercount),
    fullnamewords, 
    (fullnamenumericalcount / fullnamecharectercount),
    unameequalsfullname,
    biolength,
    exturl, 
    private, 
    postcount, 
    followers, 
    following]])

print(pred)