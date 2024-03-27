from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from io import BytesIO
import pickle
from bs4 import BeautifulSoup
import requests
import cv2
import os
import tempfile

app = Flask(__name__)

pickle_model = pickle.load(open('skin_cancer_model.pkl', 'rb'))

lung_model = pickle.load(open('lung_cancer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about.html')
def about():
    return render_template('about.html')


@app.route('/signup.html')
def signup():
    return render_template('signup.html')

@app.route('/skin.html')
def skin():
    return render_template('skin.html')


@app.route('/result.html', methods=['POST'])
def predict():
    classes = {4: ('nv', ' melanocytic nevi'), # working
               6: ('mel', 'melanoma'), # link issue
               2: ('bkl', 'benign keratosis-like lesions'),  # link issue
               1: ('bcc', ' basal cell carcinoma'),   # working
               5: ('vasc', ' pyogenic granulomas and hemorrhage'),
               0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
               3: ('df', 'dermatofibroma')}

    links_skin = {
        0: 'https://www.mayoclinic.org/diseases-conditions/actinic-keratosis/symptoms-causes/syc-20354969',
        1: 'https://www.mayoclinic.org/diseases-conditions/basal-cell-carcinoma/symptoms-causes/syc-20354187',
        2: 'https://www.mayoclinic.org/diseases-conditions/jaw-tumors-cysts/symptoms-causes/syc-20350973',
        3: 'https://dermnetnz.org/topics/dermatofibroma',
        4: 'https://emedicine.medscape.com/article/1058445-overview?form=fpf',
        5: 'https://www.healthline.com/health/pyogenic-granuloma',
        6: 'https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884'
    }

    classes_for_skinws = {
        0: 'cmp-text__rich-content cmp-dita-content cmp-text--body-sans-medium',
        1: 'cmp-text__rich-content cmp-dita-content cmp-text--body-sans-medium',
        2: 'content',
        3: '[ js-main-content ]',
        4: 'refsection_content',
        5: 'article-body css-d2znx6 undefined',
        6: 'content'
    }

    if request.method == 'POST':
        img = request.files['image']
        # image = Image.open(BytesIO())
        image = Image.open(BytesIO(img.read()))
        image = image.resize((28, 28))
        img = np.array(image)
        img = img.reshape(-1, 28, 28, 3)
        result = pickle_model.predict(img)
        # print(result[0])
        result = result.tolist()
        max_prob = max(result[0])
        class_ind = result[0].index(max_prob)
        # print(classes[class_ind])
        line = ""
        source = requests.get(links_skin[class_ind]).text
        soup = BeautifulSoup(source, 'html.parser')
        for headline in soup.findAll('div',class_=classes_for_skinws[class_ind]):
            line = headline.text

        return render_template('result.html', class_result=classes[class_ind], content=line)


@app.route('/lungs.html')
def lungs():
    return render_template('lungs.html')


@app.route('/result_lungs.html', methods=['POST'])
def result():
    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_image = request.files['image']

        if uploaded_image:
            # Create a temporary file to save the uploaded image
            temp_image = tempfile.NamedTemporaryFile(delete=False)
            uploaded_image.save(temp_image.name)

            classes = {
                0: ('colon_aca', 'colon adenocarcinoma'), ##link issue
                1: ('colon_n', 'colon benign tissue'),
                2: ('lung_aca', 'lung adenocarcinoma'), ##working
                3: ('lung_n', 'lung benign tissue'), ##working
                4: ('lung_scc', 'lung squamous cell carcinoma') ##working
            }

            links = {
                0: 'https://www.mayoclinic.org/diseases-conditions/colon-cancer/symptoms-causes/syc-20353669',
                1: 'https://www.medstarhealth.org/services/benign-tumors-of-the-colon-and-rectum',
                2: 'https://www.cancercenter.com/cancer-types/lung-cancer/types/adenocarcinoma-of-the-lung',
                3: 'https://my.clevelandclinic.org/health/diseases/15023-benign-lung-tumors',
                4: 'https://www.health.harvard.edu/cancer/squamous-cell-carcinoma-of-the-lung'
            }

            class_for_ws = {
                0: 'content',
                1: 'component promo col-12',
                2: 'component rich-text margin-bottom-standard',
                3: 'scroll-mt-[112px]',
                4: 'content-repository-content prose max-w-md-lg mx-auto flow-root getShouldDisplayAdsAttribute'
            }

            IMG_SIZE = 256
            X = []
            img = cv2.imread(temp_image.name)  # Read the image from the temporary file
            X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))

            X = np.array(X)
            # Make predictions
            result = lung_model.predict(X)
            result = result.tolist()
            max_prob = max(result[0])
            class_ind = result[0].index(max_prob)

            line=""

            source = requests.get(links[class_ind]).text
            soup = BeautifulSoup(source,'html.parser')
            for headline in soup.findAll('div', class_=class_for_ws[class_ind]):
                line = headline.text

            return render_template('result_lungs.html', class_result=classes[class_ind],content=line)

    # Handle the case when no image is uploaded or an error occurs
    return "No image uploaded or an error occurred."


if __name__ == '__main__':
    app.run(debug=True)




