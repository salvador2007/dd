from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import hashlib
from datetime import datetime
import tempfile
import logging

# DeepFace 임포트를 try-catch로 감싸서 에러 처리
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not available. Face analysis will use mock data.")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key_change_this_in_production')

# 설정
DB_PATH = 'analysis_data.db'
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin1234')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 업로드 폴더 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 한글 폰트 설정 (에러 방지)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def init_db():
    """데이터베이스 초기화"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            gender_confidence REAL,
            emotion TEXT,
            emotion_confidence REAL,
            emotion_scores TEXT,
            genres TEXT,
            filename_hash TEXT,
            face_shape TEXT DEFAULT 'Unknown'
        )
    ''')
    conn.commit()
    conn.close()

def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_hash(file_content):
    """파일 해시 생성"""
    return hashlib.md5(file_content).hexdigest()[:12]

def analyze_face(image_path):
    """얼굴 분석 함수 (DeepFace 사용 또는 모의 데이터)"""
    if not DEEPFACE_AVAILABLE:
        # DeepFace가 없을 때 모의 데이터 반환
        import random
        emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
        genders = ['Man', 'Woman']
        
        dominant_emotion = random.choice(emotions)
        emotion_scores = {emotion: random.uniform(0, 100) for emotion in emotions}
        emotion_scores[dominant_emotion] = max(emotion_scores[dominant_emotion], 70)
        
        return {
            'age': random.randint(18, 65),
            'gender': random.choice(genders),
            'gender_confidence': random.uniform(70, 95),
            'emotion': dominant_emotion,
            'emotion_confidence': emotion_scores[dominant_emotion],
            'emotion_scores': emotion_scores
        }
    
    try:
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False
        )
        
        # 결과가 리스트인 경우 첫 번째 요소 사용
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        age = int(analysis.get('age', 25))
        gender = analysis.get('gender', 'Unknown')
        emotion = analysis.get('dominant_emotion', 'neutral')
        emotion_scores = analysis.get('emotion', {'neutral': 100})
        
        # 성별 처리
        gender_confidence = 85.0
        if isinstance(gender, dict):
            gender_confidence = max(gender.values())
            gender = max(gender, key=gender.get)
        
        # 감정 신뢰도
        emotion_confidence = emotion_scores.get(emotion, 0)
        
        return {
            'age': age,
            'gender': gender,
            'gender_confidence': gender_confidence,
            'emotion': emotion,
            'emotion_confidence': emotion_confidence,
            'emotion_scores': emotion_scores
        }
    except Exception as e:
        app.logger.error(f"Face analysis error: {e}")
        # 에러 시 기본값 반환
        return {
            'age': 25,
            'gender': 'Unknown',
            'gender_confidence': 0,
            'emotion': 'neutral',
            'emotion_confidence': 0,
            'emotion_scores': {'neutral': 100}
        }

@app.errorhandler(404)
def page_not_found(e):
    """404 에러 핸들러"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """500 에러 핸들러"""
    return render_template('500.html'), 500

@app.route('/health')
def health_check():
    """헬스 체크 엔드포인트"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """파일 업로드 및 분석"""
    if 'photo' not in request.files:
        flash('파일이 선택되지 않았습니다.', 'error')
        return redirect(url_for('index'))
    
    file = request.files['photo']
    if file.filename == '' or not allowed_file(file.filename):
        flash('PNG, JPG, JPEG 파일만 업로드 가능합니다.', 'error')
        return redirect(url_for('index'))
    
    try:
        # 파일 읽기 및 해시 생성
        file_content = file.read()
        file_hash = get_file_hash(file_content)
        
        # 임시 파일 생성 및 분석
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        # 얼굴 분석
        analysis_result = analyze_face(temp_path)
        
        # 임시 파일 삭제
        os.unlink(temp_path)
        
        # 선택된 장르 처리
        selected_genres = request.form.getlist('genre')
        genres_str = ', '.join(selected_genres)
        
        # 데이터베이스 저장
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO analysis_results 
            (timestamp, age, gender, gender_confidence, emotion, emotion_confidence, 
             emotion_scores, genres, filename_hash, face_shape)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            analysis_result['age'],
            analysis_result['gender'],
            analysis_result['gender_confidence'],
            analysis_result['emotion'],
            analysis_result['emotion_confidence'],
            str(analysis_result['emotion_scores']),
            genres_str,
            file_hash,
            'Unknown'
        ))
        conn.commit()
        conn.close()
        
        return render_template('result.html',
            age=analysis_result['age'],
            gender=analysis_result['gender'],
            gender_confidence=analysis_result['gender_confidence'],
            emotion=analysis_result['emotion'],
            confidence=analysis_result['emotion_confidence'],
            emotion_scores=analysis_result['emotion_scores'],
            selected_genres=selected_genres)
    
    except Exception as e:
        app.logger.error(f'파일 처리 중 오류: {str(e)}')
        flash(f'파일 처리 중 오류가 발생했습니다: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """관리자 로그인"""
    if request.method == 'POST':
        password = request.form.get('password')
        if password == ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('admin'))
        else:
            flash('비밀번호가 올바르지 않습니다.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    """로그아웃"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    """관리자 페이지"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, age, gender, emotion, genres, filename_hash 
            FROM analysis_results ORDER BY timestamp DESC LIMIT 50
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        # HTML 테이블 생성
        table_html = '''
        <table class="data-table">
            <thead>
                <tr>
                    <th>시간</th>
                    <th>나이</th>
                    <th>성별</th>
                    <th>감정</th>
                    <th>선호 장르</th>
                    <th>파일 해시</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for row in rows:
            table_html += f'''
                <tr>
                    <td>{row[0]}</td>
                    <td>{row[1]}</td>
                    <td>{row[2]}</td>
                    <td>{row[3]}</td>
                    <td>{row[4]}</td>
                    <td>{row[5]}</td>
                </tr>
            '''
        
        table_html += '</tbody></table>'
        
        return render_template('admin.html', table_html=table_html)
    
    except Exception as e:
        app.logger.error(f'데이터 로드 중 오류: {str(e)}')
        flash(f'데이터 로드 중 오류: {str(e)}', 'error')
        return render_template('admin.html', table_html='<p>데이터를 불러올 수 없습니다.</p>')

@app.route('/graph')
def graph():
    """통계 그래프 페이지"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()
        
        if df.empty:
            return render_template('graph.html', 
                                 genre_plot="", 
                                 emotion_plot="", 
                                 face_plot="")
        
        # 장르 분포 그래프
        genre_plot = create_genre_plot(df)
        
        # 감정별 장르 분포 그래프
        emotion_plot = create_emotion_plot(df)
        
        # 얼굴형별 장르 분포 그래프 (데이터가 있는 경우)
        face_plot = create_face_plot(df)
        
        return render_template('graph.html',
                             genre_plot=genre_plot,
                             emotion_plot=emotion_plot,
                             face_plot=face_plot)
    
    except Exception as e:
        app.logger.error(f"Graph generation error: {e}")
        return render_template('graph.html', 
                             genre_plot="", 
                             emotion_plot="", 
                             face_plot="")

def create_genre_plot(df):
    """장르 분포 그래프 생성"""
    try:
        # 장르 데이터 처리
        genre_data = []
        for genres_str in df['genres'].dropna():
            if genres_str:
                genre_data.extend([g.strip() for g in genres_str.split(',')])
        
        if not genre_data:
            return ""
        
        genre_counts = pd.Series(genre_data).value_counts()
        
        plt.figure(figsize=(10, 6))
        genre_counts.plot(kind='bar', color='skyblue')
        plt.title('장르별 선호도', fontsize=14)
        plt.xlabel('장르')
        plt.ylabel('선택 횟수')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        
        return plot_data
    except Exception as e:
        app.logger.error(f"Genre plot error: {e}")
        return ""

def create_emotion_plot(df):
    """감정별 장르 분포 그래프 생성"""
    try:
        # 감정과 장르 데이터 결합
        emotion_genre_data = []
        for _, row in df.iterrows():
            if pd.notna(row['emotion']) and pd.notna(row['genres']):
                genres = [g.strip() for g in row['genres'].split(',')]
                for genre in genres:
                    emotion_genre_data.append({'emotion': row['emotion'], 'genre': genre})
        
        if not emotion_genre_data:
            return ""
        
        emotion_df = pd.DataFrame(emotion_genre_data)
        crosstab = pd.crosstab(emotion_df['emotion'], emotion_df['genre'])
        
        plt.figure(figsize=(12, 8))
        crosstab.plot(kind='bar', stacked=True)
        plt.title('감정별 장르 선호도', fontsize=14)
        plt.xlabel('감정')
        plt.ylabel('선택 횟수')
        plt.xticks(rotation=45)
        plt.legend(title='장르', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        
        return plot_data
    except Exception as e:
        app.logger.error(f"Emotion plot error: {e}")
        return ""

def create_face_plot(df):
    """얼굴형별 장르 분포 그래프 생성"""
    try:
        # 얼굴형 데이터가 있는지 확인
        face_data = df[df['face_shape'] != 'Unknown']
        if face_data.empty:
            return ""
        
        # 얼굴형과 장르 데이터 결합
        face_genre_data = []
        for _, row in face_data.iterrows():
            if pd.notna(row['face_shape']) and pd.notna(row['genres']):
                genres = [g.strip() for g in row['genres'].split(',')]
                for genre in genres:
                    face_genre_data.append({'face_shape': row['face_shape'], 'genre': genre})
        
        if not face_genre_data:
            return ""
        
        face_df = pd.DataFrame(face_genre_data)
        crosstab = pd.crosstab(face_df['face_shape'], face_df['genre'])
        
        plt.figure(figsize=(12, 8))
        crosstab.plot(kind='bar', stacked=True)
        plt.title('얼굴형별 장르 선호도', fontsize=14)
        plt.xlabel('얼굴형')
        plt.ylabel('선택 횟수')
        plt.xticks(rotation=45)
        plt.legend(title='장르', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        
        return plot_data
    except Exception as e:
        app.logger.error(f"Face plot error: {e}")
        return ""

@app.route('/recommend')
def recommend():
    """장르 추천 페이지"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()
        
        if df.empty:
            return render_template('recommend.html', 
                                 recommendations=[], 
                                 insights=[], 
                                 total_users=0)
        
        # 감정별 장르 추천 생성
        recommendations = generate_recommendations(df)
        
        # 데이터 인사이트 생성
        insights = generate_insights(df)
        
        return render_template('recommend.html',
                             recommendations=recommendations,
                             insights=insights,
                             total_users=len(df))
    
    except Exception as e:
        app.logger.error(f"Recommendation error: {e}")
        return render_template('recommend.html', 
                             recommendations=[], 
                             insights=[], 
                             total_users=0)

def generate_recommendations(df):
    """감정별 장르 추천 생성"""
    try:
        recommendations = []
        
        # 감정별로 그룹화하여 가장 인기있는 장르 찾기
        for emotion in df['emotion'].unique():
            if pd.isna(emotion):
                continue
            
            emotion_data = df[df['emotion'] == emotion]
            genre_counts = {}
            
            for genres_str in emotion_data['genres'].dropna():
                if genres_str:
                    genres = [g.strip() for g in genres_str.split(',')]
                    for genre in genres:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            if genre_counts:
                top_genre = max(genre_counts, key=genre_counts.get)
                count = genre_counts[top_genre]
                percentage = (count / len(emotion_data)) * 100
                
                recommendations.append({
                    'emotion': emotion,
                    'genre': top_genre,
                    'count': count,
                    'percentage': percentage
                })
        
        return sorted(recommendations, key=lambda x: x['percentage'], reverse=True)
    
    except Exception as e:
        app.logger.error(f"Recommendation generation error: {e}")
        return []

def generate_insights(df):
    """데이터 인사이트 생성"""
    try:
        insights = []
        
        # 기본 통계
        total_users = len(df)
        insights.append(f"총 {total_users}명의 사용자가 분석을 완료했습니다.")
        
        # 가장 많은 감정
        if not df['emotion'].isna().all():
            most_common_emotion = df['emotion'].value_counts().index[0]
            emotion_count = df['emotion'].value_counts().iloc[0]
            insights.append(f"가장 많이 감지된 감정은 '{most_common_emotion}'입니다. ({emotion_count}명)")
        
        # 평균 나이
        if not df['age'].isna().all():
            avg_age = df['age'].mean()
            insights.append(f"사용자 평균 나이는 {avg_age:.1f}세입니다.")
        
        # 성별 분포
        if not df['gender'].isna().all():
            gender_counts = df['gender'].value_counts()
            if len(gender_counts) > 0:
                top_gender = gender_counts.index[0]
                gender_percentage = (gender_counts.iloc[0] / total_users) * 100
                insights.append(f"사용자의 {gender_percentage:.1f}%가 {top_gender}으로 분석되었습니다.")
        
        return insights
    
    except Exception as e:
        app.logger.error(f"Insights generation error: {e}")
        return []

@app.route('/correlation')
def correlation():
    """상관관계 분석 페이지"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()
        
        # 필요한 열만 선택하고 결측치 제거
        cols = ["age", "emotion", "face_shape", "genres"]
        df_clean = df[cols].dropna()
        
        if df_clean.empty:
            return render_template("correlation.html", correlation_plot="")
        
        # 범주형 변수 원-핫 인코딩
        df_encoded = pd.get_dummies(df_clean, columns=["emotion", "face_shape", "genres"])
        
        # 상관계수 계산
        corr = df_encoded.corr().round(2)
        
        # 히트맵 그리기
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", cbar=True)
        plt.title("얼굴형/감정/장르 상관관계 히트맵", fontsize=14)
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        
        return render_template("correlation.html", correlation_plot=plot_data)
    
    except Exception as e:
        app.logger.error(f"Correlation error: {e}")
        return render_template("correlation.html", correlation_plot="")

if __name__ == '__main__':
    # 데이터베이스 초기화
    init_db()
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 개발 서버 실행
    app.run(debug=True, host='0.0.0.0', port=5000)