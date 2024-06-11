from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from markupsafe import Markup

# Initialize the Flask application
app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/eye_disease_prediction'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the pre-trained Keras models
model_1 = load_model('mobile-netv2.h5')
model_2 = load_model('mobilenetV1_model_tensorflowNew.h5')
model_3 = load_model('mobilenetv2.h5')

# Define a function to load and preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale image
    return img_array

# Dictionary with explanations for each condition
condition_explanations = {
    'cataract': Markup('''   <h1>Katarak</h1>
    <p>Katarak adalah kondisi di mana lensa mata menjadi keruh, menyebabkan penglihatan menjadi kabur atau buram. Ini adalah salah satu penyebab umum gangguan penglihatan pada orang dewasa.</p>

    <h1>Penyebab Katarak</h1>
    <p>Katarak terjadi ketika protein dalam lensa mata menggumpal dan membuat lensa menjadi keruh. Beberapa faktor risiko untuk katarak meliputi:</p>
    
        <p>◉ Penuaan</p>
        <p>◉ Paparan sinar matahari berlebihan</p>
        <p>◉ Riwayat keluarga</p>
        <p>◉ Konsumsi alkohol dan merokok</p>
        <p>◉ Penggunaan steroid jangka panjang</p>
        <p>◉ Penyakit seperti diabetes</P>
   

    <h1>Gejala Katarak</h1>
    <p>Gejala katarak bisa bervariasi, tetapi beberapa gejala umum termasuk:</p>
   
        <p>◉Penglihatan kabur atau buram</p>
        <p>◉Penglihatan ganda</p>
        <p>◉Penglihatan warna yang pudar</p>
        <p>◉Kesulitan melihat di malam hari</p>
        <p>◉Penglihatan yang terganggu oleh cahaya terang</p>
    

    <h1>Penanganan Katarak</h1>
    <p>Penanganan katarak melibatkan pembedahan untuk mengganti lensa yang keruh dengan lensa buatan yang jernih. Prosedur ini disebut sebagai operasi katarak atau facoemulsifikasi. Langkah-langkah penanganan katarak meliputi:</p>
    
        <p>◉Evaluasi Mata: Dokter mata akan mengevaluasi kondisi mata dan memeriksa kesehatan umum sebelum memutuskan apakah pembedahan diperlukan.</p>
        <p>◉Pembedahan: Selama operasi katarak, lensa yang keruh dihilangkan dan diganti dengan lensa buatan yang disebut implank intraokular.</p>
        <p>◉Rehabilitasi Pascaoperasi: Setelah operasi, pasien mungkin perlu menggunakan tetes mata dan mengikuti instruksi dokter mata untuk pemulihan yang cepat.</p>


    <h1>Pencegahan</h1>
    <p>Beberapa langkah yang dapat membantu mencegah atau menunda perkembangan katarak meliputi:</p>
   
        <p>◉Memakai kacamata hitam untuk melindungi mata dari sinar UV</p>
        <p>◉Menghindari merokok</p>
        <p>◉Mengontrol diabetes dan kondisi kesehatan lainnya</p>
        <p>◉Mengonsumsi makanan sehat yang kaya antioksidan seperti buah dan sayuran</p>
        <p>◉Menjaga berat badan sehat</p>
    '''),
    'retinopathy': Markup('''<h1>Diabetic Retinopathy</h1>
    <p>Diabetic retinopathy adalah komplikasi diabetes yang memengaruhi mata. Kondisi ini terjadi ketika tingginya kadar gula darah menyebabkan kerusakan pada pembuluh darah kecil di retina, yaitu jaringan sensitif cahaya yang terletak di bagian belakang mata. Diabetic retinopathy dapat menyebabkan pembengkakan, kebocoran, atau bahkan pertumbuhan pembuluh darah baru yang abnormal pada retina, yang pada akhirnya bisa menyebabkan kebutaan jika tidak diobati.</p>

    <h1>Tahapan Diabetic Retinopathy</h1>
    
        <p>◉Retinopati Non-Proliferatif (Non-Proliferative Diabetic Retinopathy - NPDR)</p>
        <p>◉Retinopati Proliferatif (Proliferative Diabetic Retinopathy - PDR)</p>


    <h1>Penanganan Diabetic Retinopathy</h1>
    <p>Penanganan diabetic retinopathy bertujuan untuk memperlambat atau menghentikan perkembangan penyakit dan mencegah kebutaan. Berikut adalah beberapa metode penanganannya:</p>
    
        <p>◉Pengendalian Gula Darah</p>
        <p>◉Terapi Laser (Fotokoagulasi)</p>
        <p>◉Suntikan Obat Anti-VEGF</p>
        <p>◉Vitrektomi</p>
        <p>◉Pengendalian Faktor Risiko</p>


    <h1>Pencegahan</h1>
    <p>Pencegahan diabetic retinopathy melibatkan kontrol diabetes yang baik, termasuk:</p>
   
        <p>◉Pemantauan rutin kadar gula darah</p>
        <p>◉Pemeriksaan mata secara berkala</p>
        <p>◉Diet sehat dan olahraga teratur</p>
        <p>◉Menghindari merokok</p>
    '''),
    'glaucoma': Markup(''' <h1>Glaukoma</h1>
    <p>Glaukoma adalah sekelompok penyakit mata yang merusak saraf optik mata, yang vital untuk penglihatan yang baik. Kerusakan ini sering disebabkan oleh tekanan yang sangat tinggi di mata (tekanan intraokular). Glaukoma adalah salah satu penyebab utama kebutaan bagi orang di atas 60 tahun. Ini bisa terjadi pada usia berapa pun tetapi lebih sering terjadi pada orang dewasa yang lebih tua.</p>

    <h1>Jenis-jenis Glaukoma</h1>
    
        <p>◉Glaukoma Sudut Terbuka (Open-Angle Glaucoma)</strong>: Ini adalah bentuk glaukoma yang paling umum.</p>
        <p>◉Glaukoma Sudut Tertutup (Angle-Closure Glaucoma)</strong>: Ini adalah kondisi yang kurang umum tetapi lebih parah.</p>
        <p>◉Glaukoma Tekanan Normal (Normal-Tension Glaucoma)</strong>: Dalam jenis ini, saraf optik rusak meskipun tekanan mata dalam kisaran normal.</p>
        <p>◉Glaukoma Kongenital</strong>: Jenis ini terjadi pada bayi yang lahir dengan cacat pada sudut drainase mata yang memperlambat atau mencegah drainase normal cairan mata.</p>
        <p>◉Glaukoma Sekunder</strong>: Jenis ini terjadi akibat kondisi atau penyakit lain.</p>


    <h1>Gejala Glaukoma</h1>
    <p>Gejala glaukoma bisa bervariasi tergantung pada jenisnya:</p>
   
        <p>◉Untuk Glaukoma Sudut Terbuka: Kehilangan penglihatan dimulai dari tepi penglihatan (peripheral vision).</p>
        <p>◉Untuk Glaukoma Sudut Tertutup: Gejala bisa tiba-tiba dan parah.</p>
    

    <h1>Penanganan Glaukoma</h1>
    <p>Penanganan glaukoma berfokus pada menurunkan tekanan intraokular untuk mencegah kerusakan lebih lanjut pada saraf optik. Beberapa metode penanganan meliputi:</p>
    
        <p>◉Obat Tetes Mata</p>
        <p>◉Obat Oral</p>
        <p>◉Terapi Laser</p>
        <p>◉Pembedahan</p>


    <h1>Pencegahan</h1>
    <p>Sementara glaukoma tidak dapat selalu dicegah, beberapa langkah dapat membantu mengurangi risiko:</p>
   
        <p>◉Pemeriksaan Mata Rutin</p>
        <p>◉Mengetahui Riwayat Keluarga</p>
        <p>◉Olahraga Teratur</p>
        <p>◉Mengambil Obat Mata Sesuai Rekomendasi</p>
    '''),
    'normal': Markup(''' <h1>Mata Normal</h1>
    <p>Mata normal adalah mata yang tidak mengalami gangguan penglihatan atau penyakit mata yang serius. Ini adalah kondisi mata yang optimal, di mana semua struktur mata berfungsi dengan baik.</p>

    <h1>Fungsi Mata Normal</h1>
    <p>Mata normal memiliki kemampuan untuk:</p>
   
        <p>◉Menerima cahaya dan gambar melalui kornea dan lensa.</p>
        <p>◉Fokus gambar pada retina untuk membentuk gambar yang jelas.</p>
        <p>◉Mengirimkan sinyal visual dari retina ke otak melalui saraf optik.</p>
        <p>◉Mengatur jumlah cahaya yang masuk ke mata melalui iris.</p>
        <p>◉Menyesuaikan bentuk lensa untuk melihat objek pada jarak yang berbeda (akomodasi).</p>
    

    <h1>Pencegahan</h1>
    <p>Untuk menjaga kesehatan mata dan mencegah gangguan penglihatan, penting untuk:</p>
   
        <p>◉Memeriksakan mata secara teratur ke dokter mata untuk pemeriksaan rutin.</p>
        <p>◉Memakai kacamata atau lensa kontak jika diperlukan.</p>
        <p>◉Menghindari paparan langsung terhadap sinar UV dengan menggunakan kacamata hitam saat berada di bawah sinar matahari.</p>
        <p>◉Mengonsumsi makanan sehat yang kaya nutrisi, terutama makanan yang mengandung vitamin A, lutein, dan zeaxanthin.</p>
        <p>◉Menjaga kelembapan mata dengan menggunakan tetes mata buatan jika mata terasa kering.</p>
        <p>◉Menghindari kebiasaan merokok.</p>
        <p>◉Menjaga berat badan yang sehat dan mengelola kondisi kesehatan seperti diabetes dan tekanan darah tinggi.</p>
    ''')
}

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    upload_date = db.Column(db.DateTime, default=db.func.current_timestamp())

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Determine which model to use
        model_choice = request.form.get('model_choice')
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            img_array = preprocess_image(file_path)
            
            if model_choice == 'model_2':
                prediction = model_2.predict(img_array)
                class_labels = ['retinopathy', 'glaucoma', 'normal']
            elif model_choice == 'model_3':
                prediction = model_3.predict(img_array)
                class_labels = ['retinopathy', 'glaucoma', 'normal']
            else:
                prediction = model_1.predict(img_array)
                class_labels = ['cataract', 'retinopathy', 'glaucoma', 'normal']
            
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = class_labels[predicted_class]
            
            explanation = condition_explanations[predicted_label]
            
            new_prediction = Prediction(filename=filename, prediction=predicted_label)
            db.session.add(new_prediction)
            db.session.commit()
            
            return render_template('index.html', filename=filename, prediction=predicted_label, explanation=explanation)
    
    return render_template('index.html', filename=None, prediction=None, explanation=None)

# Route to display uploaded file
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/delete/<int:result_id>', methods=['POST'])
def delete_result(result_id):
    result = Prediction.query.get_or_404(result_id)
    db.session.delete(result)
    db.session.commit()
    return redirect(url_for('hasil'))

# Route to display results
@app.route('/hasil')
def hasil():
    results = Prediction.query.all()
    return render_template('hasil.html', results=results)

# Route to display team members
@app.route('/anggota')
def anggota():
    return render_template('anggota.html')

# Route to display homepage
@app.route('/beranda')
def beranda():
    return render_template('beranda.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=False)