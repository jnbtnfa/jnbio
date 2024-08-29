import streamlit as st
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier
import openai
import matplotlib.pyplot as plt

# OpenAI API 키 설정
openai.api_key = 'your-api-key'

# 데이터베이스 연결 및 테이블 생성
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn):
    try:
        sql_create_proteins_table = """ CREATE TABLE IF NOT EXISTS proteins (
                                            id integer PRIMARY KEY,
                                            sequence text NOT NULL,
                                            length integer,
                                            activity text
                                        ); """
        c = conn.cursor()
        c.execute(sql_create_proteins_table)
    except sqlite3.Error as e:
        print(e)

# 데이터베이스 연결
db_file = 'proteins.db'
conn = create_connection(db_file)
if conn is not None:
    create_table(conn)
else:
    st.error("Error! Cannot create the database connection.")

# GPT 설명 생성 함수
def generate_gpt_description(sequence):
    prompt = f"Provide a brief analysis of the following protein sequence: {sequence}"
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=50)
    return response.choices[0].text.strip()

# 예측 모델 학습 함수
def train_model():
    data = pd.DataFrame({
        'Sequence': ['MKQLEDKVEELLSKNYHLENEVARLKKLVGER', 'MIKESEKVEEMLSKNYHLENEVTRLKKLIGER'],
        'Length': [30, 30],
        'Activity': [1, 0]
    })
    X = data[['Length']]
    y = data['Activity']

    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# 모델 학습
model = train_model()

# Streamlit 앱 UI 구성
st.title("AI-based Protein Target Prediction")

# 사용자 입력: 단백질 서열
sequence_input = st.text_input("Enter protein sequence", "MKQLEDKVEELLSKNYHLENEVARLKKLVGER")

if st.button("Analyze"):
    sequence_length = len(sequence_input)
    prediction = model.predict([[sequence_length]])
    gpt_description = generate_gpt_description(sequence_input)
    st.write(f"Sequence Length: {sequence_length}")
    st.write(f"Predicted Activity: {'Active' if prediction[0] == 1 else 'Inactive'}")
    st.write(f"GPT Analysis: {gpt_description}")

    if st.button("Save to Database"):
        activity = 'Active' if prediction[0] == 1 else 'Inactive'
        protein_data = (sequence_input, sequence_length, activity)
        insert_protein(conn, protein_data)
        st.success("Protein sequence saved to database.")

# 파일 업로드
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    if st.button("Plot Results"):
        data['Predicted Activity'] = model.predict(data[['Length']])
        fig, ax = plt.subplots()
        ax.scatter(data['Length'], data['Predicted Activity'])
        ax.set_xlabel('Length of Sequence')
        ax.set_ylabel('Predicted Activity')
        ax.set_title('Protein Sequence Length vs Predicted Activity')
        st.pyplot(fig)
