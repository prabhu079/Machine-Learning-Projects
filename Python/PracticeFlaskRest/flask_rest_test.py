from flask import Flask, jsonify, request

app = Flask(__name__)
app.debug = True
empDB = [{'id': '101',
          'name': 'Saravanan S',
          'title': 'Technical Leader'}, {
             'id': '201',
             'name': 'Rajkumar P',
             'title': 'Sr Software Engineer'}
         ]


@app.route("/")
def home():
    return "Welcome HOME $Bengaluru"


@app.route('/empDb/employee', methods=['Get'])
def hello_world():
    return jsonify({'emps': empDB})


@app.route('/empDb/employee/<empId>', methods=['GET'])
def getEmp(empId):
    usr = [emp for emp in empDB if (emp['id'] == empId)]
    return jsonify({'emp': usr}) @ app.route('/empDb/employee', methods=['POST'])


def createEmp():
    dat = {'id': request.json['id'],
           'name': request.json['name'],
           'title': request.json['title']
           }
    empDB.append(dat)
    return jsonify(dat)


app.run(debug=True)
