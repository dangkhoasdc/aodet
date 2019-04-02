# -*- coding: utf-8 -*-
"""

"""
# std
import os
import json
import os.path as osp
import subprocess

# flask
from flask import Flask, render_template, request
from flask.json import jsonify

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf.file import FileField, FileRequired, FileAllowed

from flask_bootstrap import Bootstrap
import pandas as pd

import argparse
import pandas as pd

# aodet
import aodet.analysis as ana
import aodet.io as aio

# utils
import paramiko

app = Flask(__name__)
app.secret_key = 'dev'
imdir = osp.join(os.getcwd(), "static", "tmp")
app.config['UPLOADED_PHOTOS_DEST'] = imdir

bootstrap = Bootstrap(app)
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)
cwd = os.getcwd()


class PerformanceForm(FlaskForm):
    expid = StringField('Experiment ID', validators=[DataRequired(), Length(1, 20)])
    productside = StringField('Product Team', validators=[DataRequired(), Length(1, 20)])
    lblmapping = FileField('Label Mapping File')
    targets = FileField('Targets File')
    cfresults = FileField('Confusion Matrix Result')
    submit = SubmitField()


class OctaveDebugForm(FlaskForm):
    image = FileField(validators=[FileAllowed(photos, 'Image Only'),
                                  FileRequired('File was empty')])
    submit = SubmitField()

@app.route('/index', methods=['GET', 'POST'])
def index():
    perf_setup_form = PerformanceForm()
    if perf_setup_form.validate_on_submit():
        print("Product team: {}".format(perf_setup_form.productside.data))
        targets = pd.read_csv(request.files.get('targets'), index_col=0).to_dict('index')
        print(targets)
        # print(pd.read_csv(request.files.get('lblmapping')))
        mapping = pd.read_csv(request.files.get('lblmapping'), index_col=0, squeeze=True).to_dict()
        print(request.files.get('lblmapping'))

        results = aio.Reader.from_detection_sys(request.files.get('cfresults'),
                                                mapping)
        detana = ana.TargetAnalysis(targets, results)
        report = detana.analyze()
        prtable = get_prtable(targets, results, report)
        achieved, finetunes, underperform = get_statis(report)
        lbls = list(report.keys())
        f1scores = detana.get_f1_delta()
        suggestions, cfscls = get_suggestion(detana)
        print("confusing labels {}".format(cfscls))
        return render_template('result.html', data=prtable,
                               productteam=perf_setup_form.productside.data,
                               roc=prtable[2],
                               achieved=achieved,
                               finetunes=finetunes,
                               underperf=underperform,
                               lbls=lbls,
                               expid=perf_setup_form.expid.data,
                               f1scores=f1scores,
                               suggestions=suggestions,
                               cfscls=cfscls)
        #pr table

    return render_template('index.html', form=perf_setup_form)


def get_statis(report):
    achieved = [lbl for lbl in report.keys() if report[lbl] == ana.ResultType.ACHIEVED]
    finetunes = [lbl for lbl in report.keys() if report[lbl] == ana.ResultType.CONSIDERING]
    underperform = [lbl for lbl in report.keys() if report[lbl] == ana.ResultType.UNDERPERFORMED]
    return achieved, finetunes, underperform



def get_prtable(targets, results, detana):
    """
    return json data of targets and experiment results
    Parameters:
        :param targets: dict-like targets results
        :param results: cfm data
    """
    cfm = dict()
    roc = []
    for lbl in results.get_classes():
        cfm[lbl] = {
            'prec': results.concept_prec(lbl),
            'recall': results.concept_recall(lbl),
            'ret': detana[lbl]
        }
        roc.append([lbl] + list(map(lambda x: round(x, 3),
                                    results.concept_pr_point(lbl))))

    return [targets, cfm, roc]


def get_suggestion(anadet):
    ret = anadet.analyze_class()
    confusing_cls = dict()
    retsug = dict()
    print(ret)
    for lbl, v in ret.items():
        if ana.Suggestion.NORMAL in v:
            continue
        if ana.Suggestion.CONFUSING_GROUNDTRUTH in v:
            confusing_cls[lbl] = anadet.get_confusing_labels(lbl)

        retsug[lbl] = v
    return retsug, confusing_cls

    # return json.dumps([targets, cfm])
# jsonify(targets=targets, cfm=cfm)

@app.route('/perf', methods=['GET', 'POST'])
def perf():
    return render_template('index.html')

@app.route('/comp', methods=['GET', 'POST'])
def comp():
    return render_template('index.html')


def ssh_run_cmd(cmd):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy)
    client.connect('hydra2.visenze.com', username='khoa', password='KimTho1612')
    stdin, stdout, stderr = client.exec_command(cmd)
    client.close()



@app.route('/localeval', methods=['GET', 'POST'])
def localeval():
    form = OctaveDebugForm()
    if form.validate_on_submit():
        imname = photos.save(form.image.data)
        imfullpath = osp.join(imdir, imname)
        subprocess.run(['scp', ])
        # setup ssh
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.WarningPolicy)
        client.connect('hydra2.visenze.com', username='khoa', password='KimTho1612')
        sftp = client.open_sftp()
        sftp.put(imfullpath, '/mnt/ssd_01/khoa/python_scripts/detectron_tools/{}'.format(imname))
        sftp.close()
        client.close()

        # run the demo on server
        ssh_run_cmd('''cd /mnt/ssd_01/khoa/python_scripts/
                    /mnt/ssd_01/khoa/python_scripts/detectron_tools/

                    python debug_output_blobs.py -M model_final/ -I a.jpg
                    ''')
        # return render_template('local_eval.html', form=form, img_url=impath)
    return render_template('local_eval.html', form=form)

# @app.route('/form', methods=['GET', 'POST'])
# def test_form():
#     form = HelloForm()
#     return render_template('form.html', form=form)


# @app.route('/nav', methods=['GET', 'POST'])
# def test_nav():
#     return render_template('nav.html')


# @app.route('/pagination', methods=['GET', 'POST'])
# def test_pagination():
#     db.drop_all()
#     db.create_all()
#     for i in range(100):
#         m = Message()
#         db.session.add(m)
#     db.session.commit()
#     page = request.args.get('page', 1, type=int)
#     pagination = Message.query.paginate(page, per_page=10)
#     messages = pagination.items
    # return render_template('pagination.html', pagination=pagination, messages=messages)


# @app.route('/utils', methods=['GET', 'POST'])
# def test_utils():
#     return render_template('utils.html')
