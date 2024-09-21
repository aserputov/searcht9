from autocomplete import AutocompleteSystem
from flask import Flask, request, jsonify, render_template
from autocomplete import AutocompleteSystem
from db import db
app = Flask(__name__)

# Start Search
# Insert previously searched queries, if any. In our case 4 requests.
# instance = AutocompleteSystem(["what is my ip", "what time is it", "why is the rain", "does is it means"], [1,2,2,4])


print(db)

instance = AutocompleteSystem(db[0], db[1])

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    
    query = request.args.get('q', '') 
    # print(query)
    suggestions = instance.input(query)
    instance.input('#')
    # print(suggestions)
    return jsonify(suggestions)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


# Start new search. Instead of Enter use # 
# print(instance.input("w"))
# print(instance.input("#"))

# print(instance.input("wha"))
# print(instance.input("#"))

# print(instance.input("why"))
# print(instance.input("#"))
# # Start new search, notice how w will be suggested for any new requests starting with w
# print(instance.input("w"))
# print(instance.input("#"))


# print(instance.input("wh"))
# print(instance.input("#"))

# print(instance.input("why"))