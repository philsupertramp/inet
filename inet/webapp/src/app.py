import json

import numpy as np
from flask import Flask, render_template, request

from inet.webapp.src.bounding_boxes import make_prediction
from inet.webapp.src.helper import (decode_img, get_example_files,
                                    load_example_file)
from inet.webapp.src.model_manager import model_manager

app = Flask(__name__)


@app.before_first_request
def load_models():
    """load models after startup to allows cold start, first request will take some time."""
    model_manager.load_models()


@app.route('/')
def home():
    """home route"""
    return render_template('home.html')


@app.route('/bounding-boxes/', methods=['GET', 'POST'])
def bounding_boxes():
    """Route to demonstrate bounding box regression with different models"""
    prediction = None
    files = get_example_files()
    models = set(model_manager.get_all_models()) - {'default'}

    # select current model name
    current_model = request.args.get('model')
    if current_model:
        model_manager.select(current_model)
    else:
        current_model = model_manager.get().model_name

    render_kwargs = {
        'files': files, 'models': models, 'template_name_or_list': 'bounding_boxes.html',
        'selected_model': current_model
    }

    # handle task request
    if file_name := request.args.get('file', 'custom' if request.method == 'POST' else None):
        # handle custom file upload
        if file_name == 'custom':
            if request.method != 'POST':
                return render_template(**{**render_kwargs, 'error': 'No image provided.'})

            # preprocess image
            content = request.form['customFileContent']
            img = np.array(decode_img(content).resize((224, 224)))
            prediction = {'src': make_prediction(img, model_manager.get()), 'name': 'custom'}
        else:
            # handle selected file
            prediction = {'src': make_prediction(load_example_file(file_name), model_manager.get()), 'name': file_name}

    return render_template(
        prediction=prediction,
        **render_kwargs
    )


@app.route('/available-models')
def available_models():
    """Route to list all available models"""
    models = set(model_manager.get_all_models()) - {'default'}
    with open('./model-weights/content.json') as f:
        cfgs = json.load(f)
    output = []
    for cfg in cfgs:
        if cfg.get('name') in models:
            output.append(cfg)

    return render_template('models.html', configs=output)
