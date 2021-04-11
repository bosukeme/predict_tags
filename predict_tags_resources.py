from flask_restful import Resource, reqparse
from predict_tags import process_all_predictions


class PredictTags(Resource):
    def get(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('text', type=str, help="The text/complains of the client")
            
            args = parser.parse_args()



            result = process_all_predictions(args['text']) 
            return {
                'status': 'success',
                'data': result, 
                'message': 'Tags prediction successful.'
            }, 200

        except Exception as e:
            return {
                'status': 'failed',
                'data': None,
                'message': str(e)
            }, 500

