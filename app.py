from flask import Flask, render_template, request
import joblib
import pandas as pd
import pyodbc
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from flask import send_file

app = Flask(__name__)

model = joblib.load("models/student_risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
ordinal_mappings = joblib.load("models/ordinal_mappings.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

explainer = shap.TreeExplainer(model)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        student_id = request.form["student_id"]
        input_data = {}

        for col in feature_columns:
            input_data[col] = int(request.form[col])

        input_df = pd.DataFrame([input_data])

        input_scaled = scaler.transform(input_df)

        input_scaled_df = pd.DataFrame(
            input_scaled,
            columns=input_df.columns
        )

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        confidence = round(max(probability) * 100, 2)
        result = "At Risk" if prediction == 1 else "Not At Risk"

        # Generate SHAP explanation
        shap_values = explainer(input_scaled_df)

        plt.figure(figsize=(10, 6))

        shap.plots.waterfall(
            shap_values[0],
            max_display=10,
            show=False
        )

        plot_path = os.path.join("static", "shap_plot.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        feature_names = input_df.columns.tolist()
        shap_vals = shap_values.values[0]

        feature_impacts = list(zip(feature_names, shap_vals))

        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        top_positive = []
        top_negative = []

        for feature, value in feature_impacts:
            if value > 0 and len(top_positive) < 3:
                top_positive.append(f"{feature} increased risk")
            elif value < 0 and len(top_negative) < 3:
                top_negative.append(f"{feature} reduced risk")

        risk_factors = top_positive
        protective_factors = top_negative

        # SQL Server connection
        conn = pyodbc.connect(
            r'DRIVER={SQL Server};'
            r'SERVER=DESKTOP-8D7CJ5E\SQLEXPRESS;'
            r'DATABASE=StudentRiskDB;'
            r'Trusted_Connection=yes;'
        )

        cursor = conn.cursor()

        insert_query = """
        INSERT INTO PredictionLogs (
            Student_ID,Hours_Studied, Attendance, Parental_Involvement,
            Access_to_Resources, Extracurricular_Activities, Sleep_Hours,
            Previous_Scores, Motivation_Level, Internet_Access,
            Tutoring_Sessions, Family_Income, Teacher_Quality,
            School_Type, Peer_Influence, Physical_Activity,
            Learning_Disabilities, Parental_Education_Level,
            Distance_from_Home, Gender,
            Prediction_Result, Confidence
        )
        VALUES (?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        values = [student_id] + list(input_data.values()) + [result, confidence]

        cursor.execute(insert_query, values)
        conn.commit()

        cursor.close()
        conn.close()

        global latest_report_data

        latest_report_data = {
            "student_id": student_id,
            "prediction": result,
            "confidence": confidence,
            "input_data": input_data,
            "risk_factors": risk_factors,
            "protective_factors": protective_factors
        }

        return render_template(
            "index.html",
            prediction=result,
            confidence=confidence,
            shap_plot=True,
            risk_factors=risk_factors,
            protective_factors=protective_factors
        )

    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/history")
def history():

    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=DESKTOP-8D7CJ5E\\SQLEXPRESS;"
        "DATABASE=StudentRiskDB;"
        "Trusted_Connection=yes;"
    )

    cursor = conn.cursor()

    cursor.execute("""
        SELECT TOP 20
            Id,
            Student_ID,
            Prediction_Result,
            Confidence,
            Prediction_Time
        FROM PredictionLogs
        ORDER BY Prediction_Time DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return render_template("history.html", rows=rows)

@app.route("/compare", methods=["GET", "POST"])
def compare():

    if request.method == "POST":

        student_id = request.form["student_id"]

        conn = pyodbc.connect(
            "DRIVER={SQL Server};"
            "SERVER=DESKTOP-8D7CJ5E\\SQLEXPRESS;"
            "DATABASE=StudentRiskDB;"
            "Trusted_Connection=yes;"
        )

        cursor = conn.cursor()

        cursor.execute("""
                       SELECT *
                       FROM PredictionLogs
                       WHERE Student_ID = ?
                       ORDER BY Prediction_Time DESC
                       """, student_id)

        rows = cursor.fetchall()
        comparison = None
        message = None
        no_records = False

        if len(rows) == 0:
            no_records=True
        elif len(rows) == 1:
            message = "Only one prediction record found. At least two predictions are required for comparison."

        if len(rows) >= 2:

            latest = rows[0]
            previous = rows[1]
            numeric_changes = []

            features_to_compare = ["Hours_Studied","Attendance","Sleep_Hours","Previous_Scores",
                "Tutoring_Sessions","Physical_Activity"]
            feature_display_names = {
                "Hours_Studied": "Hours Studied",
                "Attendance": "Attendance",
                "Sleep_Hours": "Sleep Hours",
                "Previous_Scores": "Previous Scores",
                "Tutoring_Sessions": "Tutoring Sessions",
                "Physical_Activity": "Physical Activity"
            }
            for feature in features_to_compare:
                latest_value = getattr(latest, feature)
                previous_value = getattr(previous, feature)

                change = abs(latest_value - previous_value)

                numeric_changes.append({
                    "feature": feature_display_names[feature],
                    "old": previous_value,
                    "new": latest_value,
                    "difference": change
                })

            numeric_changes.sort(
                key=lambda x: x["difference"],
                reverse=True
            )

            top_changes = numeric_changes[:3]

            confidence_change = round(
                latest.Confidence - previous.Confidence,
                2
            )
            if confidence_change > 0:
                confidence_class = "positive-change"
            elif confidence_change < 0:
                confidence_class = "negative-change"
            else:
                confidence_class = "neutral-change"
            if confidence_change > 0:
                confidence_change_display = f"+{confidence_change}%"
            else:
                confidence_change_display = f"{confidence_change}%"

            if (previous.Prediction_Result == "At Risk" and
                    latest.Prediction_Result == "Not At Risk"):
                analysis = "Student performance has improved compared to the previous prediction."

            elif ( previous.Prediction_Result == "Not At Risk" and
                   latest.Prediction_Result == "At Risk"):
                analysis = "Student performance has declined compared to the previous prediction."

            else:
                analysis = "Student performance remains relatively stable."

            comparison = {
                "previous_result": previous.Prediction_Result,
                "latest_result": latest.Prediction_Result,
                "previous_confidence": previous.Confidence,
                "latest_confidence": latest.Confidence,
                "confidence_change": confidence_change,
                "confidence_change_display": confidence_change_display,
                "confidence_class": confidence_class,
                "analysis": analysis,
                "top_changes": top_changes
            }

        conn.close()

        return render_template(
            "compare.html",
            rows=rows,
            student_id=student_id,
            comparison=comparison,
            message=message,
            no_records=no_records
        )

    return render_template("compare.html")

@app.route("/download_report")
def download_report():

    global latest_report_data

    pdf_file = "student_prediction_report.pdf"

    doc = SimpleDocTemplate(pdf_file)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(
        "EduRisk - Student Performance Risk Prediction Report",
        styles['Title']
    ))

    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph(
            f"<b>Student ID:</b> {latest_report_data['student_id']}",
            styles['BodyText']
        )
    )

    elements.append(Paragraph(
        f"<b>Prediction:</b> {latest_report_data['prediction']}",
        styles['BodyText']
    ))

    elements.append(Paragraph(
        f"<b>Confidence:</b> {latest_report_data['confidence']}%",
        styles['BodyText']
    ))

    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Student Input Details</b>", styles['Heading2']))

    for key, value in latest_report_data["input_data"].items():
        elements.append(
            Paragraph(f"{key}: {value}", styles['BodyText'])
        )

    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph("<b>Factors Increasing Risk</b>", styles['Heading2'])
    )

    for factor in latest_report_data["risk_factors"]:
        elements.append(
            Paragraph(f"• {factor}", styles['BodyText'])
        )

    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph("<b>Factors Reducing Risk</b>", styles['Heading2'])
    )

    for factor in latest_report_data["protective_factors"]:
        elements.append(
            Paragraph(f"• {factor}", styles['BodyText'])
        )

    doc.build(elements)

    return send_file(pdf_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)