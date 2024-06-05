from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
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

# Load the pre-trained Keras model
model = load_model('mobilenetv2.h5')

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

    <h2>Penyebab Katarak</h2>
    <p>Katarak terjadi ketika protein dalam lensa mata menggumpal dan membuat lensa menjadi keruh. Beberapa faktor risiko untuk katarak meliputi:</p>
    <ul>
        <li>Penuaan</li>
        <li>Paparan sinar matahari berlebihan</li>
        <li>Riwayat keluarga</li>
        <li>Konsumsi alkohol dan merokok</li>
        <li>Penggunaan steroid jangka panjang</li>
        <li>Penyakit seperti diabetes</li>
    </ul>

    <h2>Gejala Katarak</h2>
    <p>Gejala katarak bisa bervariasi, tetapi beberapa gejala umum termasuk:</p>
    <ul>
        <li>Penglihatan kabur atau buram</li>
        <li>Penglihatan ganda</li>
        <li>Penglihatan warna yang pudar</li>
        <li>Kesulitan melihat di malam hari</li>
        <li>Penglihatan yang terganggu oleh cahaya terang</li>
    </ul>

    <h2>Penanganan Katarak</h2>
    <p>Penanganan katarak melibatkan pembedahan untuk mengganti lensa yang keruh dengan lensa buatan yang jernih. Prosedur ini disebut sebagai operasi katarak atau facoemulsifikasi. Langkah-langkah penanganan katarak meliputi:</p>
    <ol>
        <li>Evaluasi Mata: Dokter mata akan mengevaluasi kondisi mata dan memeriksa kesehatan umum sebelum memutuskan apakah pembedahan diperlukan.</li>
        <li>Pembedahan: Selama operasi katarak, lensa yang keruh dihilangkan dan diganti dengan lensa buatan yang disebut implank intraokular.</li>
        <li>Rehabilitasi Pascaoperasi: Setelah operasi, pasien mungkin perlu menggunakan tetes mata dan mengikuti instruksi dokter mata untuk pemulihan yang cepat.</li>
    </ol>

    <h2>Pencegahan</h2>
    <p>Beberapa langkah yang dapat membantu mencegah atau menunda perkembangan katarak meliputi:</p>
    <ul>
        <li>Memakai kacamata hitam untuk melindungi mata dari sinar UV</li>
        <li>Menghindari merokok</li>
        <li>Mengontrol diabetes dan kondisi kesehatan lainnya</li>
        <li>Mengonsumsi makanan sehat yang kaya antioksidan seperti buah dan sayuran</li>
        <li>Menjaga berat badan sehat</li>
    </ul>'''),
    'retinopathy': Markup('''<h1>Diabetic Retinopathy</h1>
    <p>Diabetic retinopathy adalah komplikasi diabetes yang memengaruhi mata. Kondisi ini terjadi ketika tingginya kadar gula darah menyebabkan kerusakan pada pembuluh darah kecil di retina, yaitu jaringan sensitif cahaya yang terletak di bagian belakang mata. Diabetic retinopathy dapat menyebabkan pembengkakan, kebocoran, atau bahkan pertumbuhan pembuluh darah baru yang abnormal pada retina, yang pada akhirnya bisa menyebabkan kebutaan jika tidak diobati.</p>

    <h2>Tahapan Diabetic Retinopathy</h2>
    <ol>
        <li>Retinopati Non-Proliferatif (Non-Proliferative Diabetic Retinopathy - NPDR)</li>
        <li>Retinopati Proliferatif (Proliferative Diabetic Retinopathy - PDR)</li>
    </ol>

    <h2>Penanganan Diabetic Retinopathy</h2>
    <p>Penanganan diabetic retinopathy bertujuan untuk memperlambat atau menghentikan perkembangan penyakit dan mencegah kebutaan. Berikut adalah beberapa metode penanganannya:</p>
    <ol>
        <li>Pengendalian Gula Darah</li>
        <li>Terapi Laser (Fotokoagulasi)</li>
        <li>Suntikan Obat Anti-VEGF</li>
        <li>Vitrektomi</li>
        <li>Pengendalian Faktor Risiko</li>
    </ol>

    <h2>Pencegahan</h2>
    <p>Pencegahan diabetic retinopathy melibatkan kontrol diabetes yang baik, termasuk:</p>
    <ul>
        <li>Pemantauan rutin kadar gula darah</li>
        <li>Pemeriksaan mata secara berkala</li>
        <li>Diet sehat dan olahraga teratur</li>
        <li>Menghindari merokok</li>
    </ul>'''),
    'glaucoma': Markup(''' <h1>Glaukoma</h1>
    <p>Glaukoma adalah sekelompok penyakit mata yang merusak saraf optik mata, yang vital untuk penglihatan yang baik. Kerusakan ini sering disebabkan oleh tekanan yang sangat tinggi di mata (tekanan intraokular). Glaukoma adalah salah satu penyebab utama kebutaan bagi orang di atas 60 tahun. Ini bisa terjadi pada usia berapa pun tetapi lebih sering terjadi pada orang dewasa yang lebih tua.</p>

    <h2>Jenis-jenis Glaukoma</h2>
    <ol>
        <li><strong>Glaukoma Sudut Terbuka (Open-Angle Glaucoma)</strong>: Ini adalah bentuk glaukoma yang paling umum.</li>
        <li><strong>Glaukoma Sudut Tertutup (Angle-Closure Glaucoma)</strong>: Ini adalah kondisi yang kurang umum tetapi lebih parah.</li>
        <li><strong>Glaukoma Tekanan Normal (Normal-Tension Glaucoma)</strong>: Dalam jenis ini, saraf optik rusak meskipun tekanan mata dalam kisaran normal.</li>
        <li><strong>Glaukoma Kongenital</strong>: Jenis ini terjadi pada bayi yang lahir dengan cacat pada sudut drainase mata yang memperlambat atau mencegah drainase normal cairan mata.</li>
        <li><strong>Glaukoma Sekunder</strong>: Jenis ini terjadi akibat kondisi atau penyakit lain.</li>
    </ol>

    <h2>Gejala Glaukoma</h2>
    <p>Gejala glaukoma bisa bervariasi tergantung pada jenisnya:</p>
    <ul>
        <li>Untuk Glaukoma Sudut Terbuka: Kehilangan penglihatan dimulai dari tepi penglihatan (peripheral vision).</li>
        <li>Untuk Glaukoma Sudut Tertutup: Gejala bisa tiba-tiba dan parah.</li>
    </ul>

    <h2>Penanganan Glaukoma</h2>
    <p>Penanganan glaukoma berfokus pada menurunkan tekanan intraokular untuk mencegah kerusakan lebih lanjut pada saraf optik. Beberapa metode penanganan meliputi:</p>
    <ol>
        <li>Obat Tetes Mata</li>
        <li>Obat Oral</li>
        <li>Terapi Laser</li>
        <li>Pembedahan</li>
    </ol>

    <h2>Pencegahan</h2>
    <p>Sementara glaukoma tidak dapat selalu dicegah, beberapa langkah dapat membantu mengurangi risiko:</p>
    <ul>
        <li>Pemeriksaan Mata Rutin</li>
        <li>Mengetahui Riwayat Keluarga</li>
        <li>Olahraga Teratur</li>
        <li>Mengambil Obat Mata Sesuai Rekomendasi</li>
    </ul>'''),
    'normal': Markup(''' <h1>Mata Normal</h1>
    <p>Mata normal adalah mata yang tidak mengalami gangguan penglihatan atau penyakit mata yang serius. Ini adalah kondisi mata yang optimal, di mana semua struktur mata berfungsi dengan baik.</p>

    <h2>Fungsi Mata Normal</h2>
    <p>Mata normal memiliki kemampuan untuk:</p>
    <ul>
        <li>Menerima cahaya dan gambar melalui kornea dan lensa.</li>
        <li>Fokus gambar pada retina untuk membentuk gambar yang jelas.</li>
        <li>Mengirimkan sinyal visual dari retina ke otak melalui saraf optik.</li>
        <li>Mengatur jumlah cahaya yang masuk ke mata melalui iris.</li>
        <li>Menyesuaikan bentuk lensa untuk melihat objek pada jarak yang berbeda (akomodasi).</li>
    </ul>

    <h2>Pencegahan</h2>
    <p>Untuk menjaga kesehatan mata dan mencegah gangguan penglihatan, penting untuk:</p>
    <ul>
        <li>Memeriksakan mata secara teratur ke dokter mata untuk pemeriksaan rutin.</li>
        <li>Memakai kacamata atau lensa kontak jika diperlukan.</li>
        <li>Menghindari paparan langsung terhadap sinar UV dengan menggunakan kacamata hitam saat berada di bawah sinar matahari.</li>
        <li>Mengonsumsi makanan sehat yang kaya nutrisi, terutama makanan yang mengandung vitamin A, lutein, dan zeaxanthin.</li>
        <li>Menjaga kelembapan mata dengan menggunakan tetes mata buatan jika mata terasa kering.</li>
        <li>Menghindari kebiasaan merokok.</li>
        <li>Menjaga berat badan yang sehat dan mengelola kondisi kesehatan seperti diabetes dan tekanan darah tinggi.</li>
    </ul>''')
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is provided
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Preprocess the image and make prediction
            img_array = preprocess_image(file_path)
            prediction = model.predict(img_array)
            
            # Decode the prediction
            predicted_class = np.argmax(prediction, axis=1)[0]
            class_labels = ['cataract', 'retinopathy', 'glaucoma', 'normal']
            predicted_label = class_labels[predicted_class]
            
            # Get the explanation for the predicted condition
            explanation = condition_explanations[predicted_label]
            
            return render_template('index.html', filename=filename, prediction=predicted_label, explanation=explanation)
    
    return render_template('index.html', filename=None, prediction=None, explanation=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/anggota')
def anggota():
    return render_template('anggota.html')

@app.route('/hasil')
def hasil():
    return render_template('hasil.html')

@app.route('/beranda')
def beranda():
    return render_template('beranda.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Disable reloader
