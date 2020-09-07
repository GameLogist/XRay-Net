from flask import render_template, request, Blueprint, flash, redirect, url_for
from flaskblog.models import Prediction
from flaskblog.main.forms import PredictForm
from flask_login import login_user, current_user, logout_user, login_required
from flaskblog.users.utils import save_picture, send_reset_email, save_prediction_picture
from flaskblog import db
from flaskblog.main.predictions import DenseNet121, HeatmapGenerator


main = Blueprint('main', __name__)


@main.route("/")
@main.route("/home")
def home():
    # page = request.args.get('page', 1, type=int)
    # posts = Post.query.order_by(Post.date_posted.desc()).paginate(page=page, per_page=5)
    return render_template('home.html')


@main.route("/about")
def about():
    return render_template('about.html', title='About')

@main.route("/predict", methods=['GET', 'POST'])
@login_required
def predict():
    form = PredictForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_prediction_picture(form.picture.data)
            image_file = url_for('static', filename='predictions/' + picture_file)
            post = Prediction(image_file=image_file, author=current_user)
            db.session.add(post)
            db.session.commit()
            # flash('Your picture has been uploaded, please wait a bit!', 'success')


            nnIsTrained = False                 
            nnClassCount = 14                  

            trBatchSize = 64
            trMaxEpoch = 3

            imgtransResize = (320, 320)
            imgtransCrop = 224

            class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
                            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
                            'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

            imgtransCrop = 224
            pathInputImage = 'E:/Minor/Code/app/12-Error-Pages/flaskblog/'+image_file
            pathInputImage2 = 'E:/Minor/Code/app/12-Error-Pages/flaskblog/'+image_file
            pathOutputImage = 'heatmap_view1_frontal.png'
            pathModel = 'E:/Minor/models/cheXpert-master/model_ones_3epoch_densenet.tar'


            h = HeatmapGenerator(pathModel, nnClassCount, imgtransCrop)

            result,labels = h.generate(pathInputImage, pathOutputImage, imgtransCrop, class_names, form.picture.data)

            print('Result Generated:', result, '\n', labels)
            ###################################

            # message = 'We predicted your disease to be ' + labels
            # flash(message, 'success')
            # result = 'E:/Minor/Code/app/12-Error-Pages/flaskblog/static/' + result
            print('Result Generated:', result, '\n', labels)

            image_file = image_file[8:]

            return render_template('result.html', title='Result', form=form, heatmap=result, original=image_file, labels=labels)
        
        else:
            flash('Upload a picture first!', 'error')
            return redirect(url_for('main.predict'))
        # flash('Please upload a valid image type.', 'danger')
    
    return render_template('predict.html', title='Predict', form=form)
