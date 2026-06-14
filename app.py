from flask import Flask, render_template, request, redirect, url_for, Response
import os
import csv
import io
import json
import sqlite3
from datetime import datetime
from difflib import SequenceMatcher
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
app.config['DATABASE'] = os.getenv('DATABASE_NAME', 'grader.db')

EURI_API_KEY = os.getenv("EURI_API_KEY")

# Hubungkan klien OpenAI mengikut ketetapan API Euron
openai_client = None
if EURI_API_KEY:
    openai_client = OpenAI(
        api_key=EURI_API_KEY,
        base_url="https://api.euron.one/api/v1/euri" )

def get_db_connection():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn


def column_exists(cursor, table_name, column_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return any(row[1] == column_name for row in cursor.fetchall())


def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_name TEXT,
                  matrix_card TEXT,
                  category TEXT,
                  score INTEGER,
                  grade_letter TEXT,
                  feedback TEXT,
                  created_at TIMESTAMP)''')

    # Tetapan skema lajur pangkalan data menggunakan gemini-2.5-flash sebagai fallback lalai
    new_columns = {
        'rubric_breakdown': 'TEXT',
        'strengths': 'TEXT',
        'weaknesses': 'TEXT',
        'suggestions': 'TEXT',
        'similarity_score': 'REAL DEFAULT 0',
        'similarity_with': 'TEXT',
        'blind_mode': 'INTEGER DEFAULT 0',
        'original_content': 'TEXT',
        'final_score': 'INTEGER',
        'manual_note': 'TEXT',
        'filename': 'TEXT',
        'model_used': 'TEXT DEFAULT "gemini-2.5-flash"'
    }
    for col, col_type in new_columns.items():
        if not column_exists(c, 'results', col):
            c.execute(f'ALTER TABLE results ADD COLUMN {col} {col_type}')
    conn.commit()
    conn.close()


init_db()


def get_grade_details(score):
    try:
        score = float(score)
    except (ValueError, TypeError):
        score = 0
    if score >= 80:
        return {'letter': 'A', 'color': '#10b981', 'label': 'Cemerlang'}
    if score >= 70:
        return {'letter': 'B', 'color': '#3b82f6', 'label': 'Baik'}
    if score >= 50:
        return {'letter': 'C', 'color': '#f59e0b', 'label': 'Memuaskan'}
    if score >= 40:
        return {'letter': 'D', 'color': '#f97316', 'label': 'Lulus'}
    return {'letter': 'F', 'color': '#ef4444', 'label': 'Gagal'}


def safe_int(value, default=0):
    try:
        return max(0, min(100, int(float(value))))
    except (ValueError, TypeError):
        return default


def safe_json_loads(value, default=None):
    if default is None:
        default = {}
    if not value:
        return default
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def calculate_similarity(new_content):
    conn = get_db_connection()
    rows = conn.execute('SELECT matrix_card, student_name, original_content FROM results WHERE original_content IS NOT NULL AND original_content != ""').fetchall()
    conn.close()

    best_score = 0
    best_match = ''
    clean_new = (new_content or '').strip().lower()
    if not clean_new:
        return 0, ''

    for row in rows:
        old_content = (row['original_content'] or '').strip().lower()
        if not old_content:
            continue
        score = SequenceMatcher(None, clean_new, old_content).ratio() * 100
        if score > best_score:
            best_score = score
            best_match = f"{row['matrix_card']} - {row['student_name'] or 'Unknown'}"
    return round(best_score, 2), best_match


def normalize_ai_data(data):
    if not isinstance(data, dict):
        data = {}
    score = safe_int(data.get('score', 0))
    breakdown = data.get('rubric_breakdown') or {}
    if not isinstance(breakdown, dict):
        breakdown = {'Overall': score}
    return {
        'score': score,
        'rubric_breakdown': breakdown,
        'feedback': str(data.get('feedback', 'No feedback generated.')),
        'strengths': str(data.get('strengths', 'Not provided.')),
        'weaknesses': str(data.get('weaknesses', 'Not provided.')),
        'suggestions': str(data.get('suggestions', 'Not provided.')),
        'writing_level': str(data.get('writing_level', 'Not detected.')),
    }


def call_euriai_sdk(system_prompt, prompt, chosen_model):
    if not openai_client:
        raise ValueError("OpenAI client is not initialized.")
        
    resp = openai_client.chat.completions.create(
        model=chosen_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=2000
    )
    return resp.choices[0].message.content


def call_euriai(prompt, chosen_model='gemini-2.5-flash'):
    if not EURI_API_KEY:
        return normalize_ai_data({
            'score': 75,
            'rubric_breakdown': {'Content': 30, 'Critical Thinking': 20, 'Structure': 15, 'Language': 10},
            'feedback': f'Demo mode ({chosen_model}): API key not found. Add EURI_API_KEY in .env to enable real AI grading.',
            'strengths': 'The submission has acceptable structure.',
            'weaknesses': 'Some explanations need more evidence.',
            'suggestions': 'Add clearer conclusions.',
            'writing_level': 'Intermediate'
        })

    system_prompt = """
You are an academic grading assistant. Return ONLY valid JSON with this exact structure:
{
  "score": 0-100,
  "rubric_breakdown": {"Content": 0-100, "Critical Thinking": 0-100, "Structure": 0-100, "Language": 0-100},
  "feedback": "short academic feedback",
  "strengths": "student strengths",
  "weaknesses": "student weaknesses",
  "suggestions": "specific improvement suggestions",
  "writing_level": "Beginner/Intermediate/Advanced"
}
Do not include markdown fences.
"""
    try:
        raw_content = call_euriai_sdk(system_prompt, prompt, chosen_model)
        clean_json = raw_content.replace('```json', '').replace('```', '').strip()
        return normalize_ai_data(json.loads(clean_json))
    except Exception as e:
        print(f'API Error (Model: {chosen_model}): {e}')
        return None


def save_result(name, matrix, category, ai_data, similarity_score, similarity_with, blind_mode, content, filename='', final_score=None, manual_note='', model_used='gemini-2.5-flash'):
    score = safe_int(final_score if final_score not in [None, ''] else ai_data.get('score', 0))
    grade = get_grade_details(score)
    conn = get_db_connection()
    conn.execute('''INSERT INTO results
        (student_name, matrix_card, category, score, grade_letter, feedback, created_at,
         rubric_breakdown, strengths, weaknesses, suggestions, similarity_score, similarity_with,
         blind_mode, original_content, final_score, manual_note, filename, model_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (name, matrix, category, score, grade['letter'], ai_data.get('feedback', ''), datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
         json.dumps(ai_data.get('rubric_breakdown', {})), ai_data.get('strengths', ''), ai_data.get('weaknesses', ''),
         ai_data.get('suggestions', ''), similarity_score, similarity_with, 1 if blind_mode else 0, content, score, manual_note, filename, model_used))
    conn.commit()
    conn.close()
    return grade


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    final_data = None
    error = ''
    grade = None
    batch_results = []

    if request.method == 'POST':
        matrix = request.form.get('matrix_card', '').strip()
        name = request.form.get('student_name', '').strip()
        category = request.form.get('category_id', '').strip()
        rubric = request.form.get('rubric', '').strip()
        content = request.form.get('content', '').strip()
        blind_mode = request.form.get('blind_mode') == 'on'
        final_score = request.form.get('final_score', '').strip()
        manual_note = request.form.get('manual_note', '').strip()
        
        chosen_model = request.form.get('model_choice', 'gemini-2.5-flash')

        uploaded_texts = []
        files = request.files.getlist('assignment_file')
        for uploaded_file in files:
            if uploaded_file and uploaded_file.filename:
                if not uploaded_file.filename.endswith('.txt'):
                    error = 'Hanya fail .txt dibenarkan.'
                    break
                try:
                    uploaded_texts.append((uploaded_file.filename, uploaded_file.read().decode('utf-8')))
                except Exception:
                    error = 'Gagal membaca fail teks. Sila paste kandungan secara manual.'
                    break

        if not error and uploaded_texts:
            for index, (filename, file_content) in enumerate(uploaded_texts, start=1):
                batch_matrix = matrix or f'BATCH-{datetime.now().strftime("%H%M%S")}-{index}'
                prompt_name = 'Anonymous Student' if blind_mode else (name or filename)
                prompt_matrix = 'Hidden for blind grading' if blind_mode else batch_matrix
                prompt = f"Student Name: {prompt_name}\nMatrix: {prompt_matrix}\nCategory: {category}\n\nAssignment Content:\n{file_content}\n\nRubric/Criteria:\n{rubric}"
                ai_data = call_euriai(prompt, chosen_model=chosen_model)
                if ai_data:
                    similarity_score, similarity_with = calculate_similarity(file_content)
                    grade = save_result(name or filename, batch_matrix, category, ai_data, similarity_score, similarity_with, blind_mode, file_content, filename, model_used=chosen_model)
                    batch_results.append({'filename': filename, 'matrix': batch_matrix, 'score': ai_data['score'], 'grade': grade, 'similarity_score': similarity_score})
            if batch_results:
                final_data = batch_results[-1]

        elif not error and matrix and content:
            prompt_name = 'Anonymous Student' if blind_mode else name
            prompt_matrix = 'Hidden for blind grading' if blind_mode else matrix
            prompt = f"Student Name: {prompt_name}\nMatrix: {prompt_matrix}\nCategory: {category}\n\nAssignment Content:\n{content}\n\nRubric/Criteria:\n{rubric}"
            final_data = call_euriai(prompt, chosen_model=chosen_model)
            if final_data:
                similarity_score, similarity_with = calculate_similarity(content)
                final_data['similarity_score'] = similarity_score
                final_data['similarity_with'] = similarity_with
                grade = save_result(name, matrix, category, final_data, similarity_score, similarity_with, blind_mode, content, final_score=final_score, manual_note=manual_note, model_used=chosen_model)
                final_data['score'] = safe_int(final_score if final_score else final_data.get('score', 0))
            else:
                error = 'Ralat memproses data dari AI. Pilihan model engine ini tiada atau menghadapi masalah.'
        elif not error:
            error = 'Sila lengkapkan No. Matrik dan isi kandungan tugasan atau muat naik fail .txt.'

    return render_template('index.html', final_data=final_data, error=error, grade=grade, batch_results=batch_results)


@app.route('/analytics')
def analytics():
    conn = get_db_connection()
    records = conn.execute('SELECT * FROM results ORDER BY created_at DESC').fetchall()
    progress_rows = conn.execute('''SELECT matrix_card, created_at, score FROM results
                                    WHERE matrix_card IS NOT NULL AND matrix_card != ''
                                    ORDER BY matrix_card, created_at ASC''').fetchall()
    conn.close()

    stats = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    total_score = 0
    pass_count = 0
    high_similarity = 0
    top_student = None

    for row in records:
        letter = row['grade_letter']
        if letter in stats:
            stats[letter] += 1
        score = row['score'] or 0
        total_score += score
        if score >= 40:
            pass_count += 1
        if (row['similarity_score'] or 0) >= 70:
            high_similarity += 1
        if top_student is None or score > (top_student['score'] or 0):
            top_student = row

    total = len(records)
    summary = {
        'total': total,
        'average': round(total_score / total, 1) if total else 0,
        'pass_rate': round((pass_count / total) * 100, 1) if total else 0,
        'high_similarity': high_similarity,
        'top_student': top_student
    }

    progress = {}
    for row in progress_rows:
        progress.setdefault(row['matrix_card'], []).append({'date': row['created_at'], 'score': row['score']})

    return render_template('analytics.html', records=records, stats=stats, summary=summary, progress=progress)


@app.route('/delete_record/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    conn = get_db_connection()
    try:
        conn.execute('DELETE FROM results WHERE id = ?', (record_id,))
        conn.commit()
    except Exception as e:
        print(f"Ralat semasa memadam rekod: {e}")
    finally:
        conn.close()
    return redirect(url_for('analytics'))


@app.route('/reset_analytics', methods=['POST'])
def reset_analytics():
    conn = get_db_connection()
    try:
        conn.execute('DELETE FROM results')
        conn.execute("DELETE FROM sqlite_sequence WHERE name='results'")
        conn.commit()
    except Exception as e:
        print(f"Ralat semasa set semula pangkalan data: {e}")
    finally:
        conn.close()
    return redirect(url_for('analytics'))


@app.route('/export_csv')
def export_csv():
    conn = get_db_connection()
    rows = conn.execute('SELECT * FROM results ORDER BY created_at DESC').fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Matrix', 'Student Name', 'Category', 'Score', 'Grade', 'Similarity', 'Similarity With', 'AI Model', 'Feedback'])
    for row in rows:
        writer.writerow([row['created_at'], row['matrix_card'], row['student_name'], row['category'], row['score'], row['grade_letter'], row['similarity_score'], row['similarity_with'], row['model_used'] or 'gemini-2.5-flash', row['feedback']])

    return Response(output.getvalue(), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=smart_grader_results.csv'})


if __name__ == '__main__':
    app.run(debug=True)