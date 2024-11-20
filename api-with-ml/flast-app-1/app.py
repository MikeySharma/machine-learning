from flask import Flask
from routes import create_routes

app = Flask(__name__)

#Register routes
create_routes(app)


if __name__ == '__main__':
    app.run(debug=True)