from flask import Flask, render_template, request
import numpy as np
import joblib
ensemble = joblib.load('model.pkl')

 
app = Flask(__name__)

@app.route("/", methods=["POST","GET"])

def index():
    if request.method == "POST":
        # Collect the attributes from the form
        attributes = [
            'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
            'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
            'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve',
            'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
            'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
            'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping',
            'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression',
            'mentality_interceptions', 'mentality_positioning', 'mentality_vision',
            'mentality_penalties', 'mentality_composure', 'defending_marking_awareness',
            'defending_standing_tackle', 'defending_sliding_tackle','goalkeeping_diving',
            'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning',
            'goalkeeping_reflexes', 'goalkeeping_speed'
        ]

        # Collect attribute values from the form
        attribute_values = [float(request.form.get(attr, 0)) for attr in attributes]

        # Calculate the mean of specified attributes
        mean_attributes = np.mean(attribute_values)

        # Prepare the data for your machine learning model
        input_data = {
            'movement_reactions': float(request.form.get('movement_reactions', 0)),
            'mentality_composure': float(request.form.get('mentality_composure', 0)),
            'passing': float(request.form.get('passing', 0)),
            'mean_attributes': mean_attributes,
            'release_clause_eur': float(request.form.get('release_clause_eur', 0)),
            'dribbling': float(request.form.get('dribbling', 0)),
            'wage_eur': float(request.form.get('wage_eur', 0)),
            'power_shot_power': float(request.form.get('power_shot_power', 0)),
            'value_eur': float(request.form.get('value_eur', 0)),
            'mentality_vision': float(request.form.get('mentality_vision', 0)),
            'attacking_short_passing': float(request.form.get('attacking_short_passing', 0)),
            'physic': float(request.form.get('physic', 0)),
            'skill_long_passing': float(request.form.get('skill_long_passing', 0)),
            'age': float(request.form.get('age', 0)),
            'shooting': float(request.form.get('shooting', 0)),
            'skill_ball_control': float(request.form.get('skill_ball_control', 0))
        }
        # Extract the values from the input_data dictionary
        input_values = list(input_data.values())

        # Use your machine learning model to make predictions
        prediction = round(ensemble.predict([input_values]))

        # Use your machine learning model to make predictions
        #prediction, confidence_level = ensemble.predict(input_data)

        return render_template("result.html", prediction=prediction)

    return render_template("form.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 