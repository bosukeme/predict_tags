from flask import Flask
from flask_restful import Api
from predict_tags_resources import PredictTags

app=Flask(__name__)

api=Api(app)


@app.route("/")
def home():
    return "<h1 style='color:blue'>This is the Tag Predictions  pipeline!</h1>"


api.add_resource(PredictTags, '/predict_tags')

if __name__=='__main__':
    app.run(port= 5000, debug=True)