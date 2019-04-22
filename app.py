# -*- coding: utf-8 -*-
"""
Monolithic design, you know me.

The real reason: I am too lazy to organize the code.
"""
# std
import os
import json
import os.path as osp
from subprocess import Popen, PIPE, list2cmdline
import glob
import cPickle as pkl
from collections import defaultdict

import numpy as np

# flask
from flask import Flask, render_template, request
from flask.json import jsonify

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FieldList, FormField, SelectField
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
# PATH INFO
cwd = os.getcwd()
imdir = osp.join(cwd, "static", "tmp")
vizdir = osp.join(cwd, "static", "viz")
embdir = osp.join(cwd, "static", "emb")
app.config['UPLOADED_PHOTOS_DEST'] = imdir

bootstrap = Bootstrap(app)
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)

# DEFAULT PARAMS
# USE FOR DEMO

MODEL_PATH = "/mnt/ssd_01/khoa/furniture_detection/jobdir/midlevel_v3.6/resnet50_retinanet_AR5_scale3/detectron_resnet_50_s500_lr0.0075_multistep_8_10/deploy/model_final/"
MODEL_UNOPT_PATH = "/mnt/ssd_01/khoa/python_scripts/detectron_tools/model_final_unopt/"
REPORT_DIR = "/mnt/ssd_01/khoa/reports/"

#########################################################
# FORM Handlers
#########################################################

T_SNE_OPTIONS = [
    ('isolated_samples', 'Isolated Samples'),
    ('subclusters', 'Subcluster Discovery'),
]

class PerformanceForm(FlaskForm):
    expid = StringField('Experiment ID', validators=[DataRequired(), Length(1, 20)])
    productside = StringField('Product Team', validators=[DataRequired(), Length(1, 20)])
    lblmapping = FileField('Label Mapping File')
    targets = FileField('Targets File')
    cfresults = FileField('Confusion Matrix Result')
    submit = SubmitField()


class OctaveDebugForm(FlaskForm):
    model_path = StringField('Model Path', default=MODEL_PATH)
    image = FileField(validators=[FileAllowed(photos, 'Image Only'),
                                  FileRequired('File was empty')])
    submit_oct = SubmitField("Run Octave Debug")


class TSNEForm(FlaskForm):
    dataset = StringField('Dataset name')
    categories = StringField('Category (All if empty)')
    tsne_options = SelectField('Discovry Mode', choices=T_SNE_OPTIONS)
    submit_tsne = SubmitField("Run T-SNE")


class ExpJsonForm(FlaskForm):
    expjson = FileField('Exp Json')
    expname = StringField('Exp Name')

class ExpJsonListForm(FlaskForm):
    category = StringField('Category', validators=[DataRequired()])
    exps = FieldList(FormField(ExpJsonForm), min_entries=3)
    submit_exp = SubmitField('Run Comparisons')

class EmbeddingSearchForm(FlaskForm):
    model_path = StringField('Model Path', default=MODEL_UNOPT_PATH)
    dataset = StringField('Dataset name')
    image = FileField(validators=[FileAllowed(photos, 'Image Only'),
                                  FileRequired('File was empty')])
    submit_emb_search = SubmitField("Run Embedding Search")

#########################################################
# ROUTE Handlers
#########################################################


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


@app.route('/comp', methods=['GET', 'POST'])
def comp():
    return render_template('comp.html')

@app.route('/localeval', methods=['GET', 'POST'])
def localeval():
    octform = OctaveDebugForm()
    tsne_form = TSNEForm()
    exp_form = ExpJsonListForm()
    emb_search_form = EmbeddingSearchForm()

    if exp_form.submit_exp.data and exp_form.validate():
        cat = exp_form.category.data
        data = []
        expnames = []
        for idx, d in enumerate(exp_form.exps.data):
            expidx = 'exps-{}-expjson'.format(idx)
            if d['expname'] is None or request.files.get(expidx) is None:
                continue

            jsondata = json.loads(
                request.files.get(expidx).read()
            )
            name = str(d['expname'])
            expnames.append(name)
            pr = find_pr_values(jsondata, cat)
            # precision = np.maximum.accumulate(pr['precision'][::-1])[::-1].tolist()
            precision = pr['precision']
            recall = pr['recall']

            for p, r in zip(precision, recall):
                data.append([name, r, p])

        return render_template('local_eval.html', form=octform,
                               tsneform=tsne_form,
                               expform = exp_form,
                               embform=emb_search_form,
                               expnames=expnames,
                               compexp=data,
                               cat=cat)


    if tsne_form.submit_tsne.data and tsne_form.validate():
        dsname = tsne_form.dataset.data
        cats = list(map(lambda s: s.strip(), tsne_form.categories.data.split(",")))
        html_viz, sos, subcluster = run_tsne_viz(dsname, cats)
        html_viz = "http://hydra2.visenze.com:4567/{}".format(osp.basename(html_viz))
        tsne_option = tsne_form.tsne_options.data
        print("tsne options: {}".format(tsne_option))
        if tsne_option == "isolated_samples":
            subcluster = None
        elif tsne_option == "subclusters":
            sos = None
        else:
            raise IOError("could not get TSNE option.")
        return render_template('local_eval.html', form=octform,
                               tsneform=tsne_form,
                               tsne_viz=html_viz,
                               sos=sos,
                               subcluster=subcluster,
                               embform=emb_search_form,
                               expform=exp_form)

    if octform.submit_oct.data and octform.validate():

        # I am lazy
        # for f in glob.glob("{}/*".format(vizdir)):
        #     print("removing {}".format(f))
        #     os.remove(f)

        imname = photos.save(octform.image.data)
        baseimname = osp.splitext(osp.basename(imname))[0]
        modelpath = octform.model_path.data
        imfullpath = osp.join(imdir, imname)
        imvizdir = osp.join(vizdir, baseimname)
        os.mkdir(imvizdir)

        process = Popen(["python",
                         "/mnt/ssd_01/khoa/python_scripts/detectron_tools/debug_output_blobs.py",
                         "-I", str(imfullpath),
                         "-M", str(modelpath),
                         "-O", str(imvizdir)],
                        stdout=PIPE)
        (output, err) = process.communicate()
        print(output)
        exit_code = process.wait()

        # done, show the result
        vizfiles = [f for f in os.listdir(imvizdir) \
                    if osp.isfile(osp.join(imvizdir, f))]

        vizfiles = get_viz_files(vizfiles, baseimname)
        print(vizfiles)
        fpn_filters = sorted(list(set([f['fpn'] for f in vizfiles])))


        return render_template('local_eval.html', form=octform,
                               vdir=baseimname,
                               vizs=vizfiles,
                               fpn_filters=fpn_filters,
                               tsneform=tsne_form,
                               embform=emb_search_form,
                               expform=exp_form)

    if emb_search_form.submit_emb_search.data and emb_search_form.validate():
        dsname = emb_search_form.dataset.data
        modelpath = emb_search_form.model_path.data
        imname = photos.save(emb_search_form.image.data)
        baseimname = osp.splitext(osp.basename(imname))[0]
        imfullpath = osp.join(imdir, imname)
        search_ret = run_embedding_search(modelpath, dsname, imfullpath)
        return render_template('local_eval.html', form=octform,
                               tsneform=tsne_form,
                               expform=exp_form,
                               embform=emb_search_form,
                               search_ret=search_ret)


    return render_template('local_eval.html', form=octform,
                           tsneform=tsne_form,
                           expform=exp_form,
                           embform=emb_search_form,
                           vizs=None,
                           compexp=None,
                           expnames=None,
                           cat=None)

#########################################################
# Backend stuff
#########################################################

def run_command(cmds):
    """
    Run an external program and wait to get the result
    Parameters:
        :param cmds: list of arguments of the command
        :return: tuple of (output string, exit code, err message)
    """
    print("command: {}".format(list2cmdline(cmds)))
    process = Popen(cmds, stdout=PIPE)
    (output, err) = process.communicate()
    print(output)
    exit_code = process.wait()
    return output, exit_code, err

def run_tsne_viz(dsname, cats):
    """
    run the script which generates the T-SNE for a detection dataset,
    return the content of HTML page
    Parameters:
        :param dataset_name: (str) dataset name
        :param cats: (list) categories
    """
    report_fpath = "{}_{}_modified_sklearn".format(dsname, "_".join(sorted(cats)))
    sos_fpath = "{}_{}_modified_sklearn_sos".format(dsname, "_".join(sorted(cats)))
    report_fullpath = osp.join(REPORT_DIR, report_fpath+".html")
    sos_fullpath = osp.join(REPORT_DIR, sos_fpath+".pkl")
    print("cats = {}".format(cats))
    if not osp.exists(report_fullpath) or not osp.exists(sos_fullpath) :
        run_command([
            "python",
            "/mnt/ssd_01/khoa/python_scripts/detection_tsne.py",
            "-I", dsname,
            "-R", report_fpath,
            "-D", "/mnt/ssd_01/khoa/furniture_detection/data",
            "-C"
        ]+list(map(str, cats)))

    # after running, apply post-process for SOS file
    sosdata = pkl.load(open(sos_fullpath, "rb"))
    postsos = defaultdict(list)
    subcluster = defaultdict(lambda: defaultdict(list))
    print("type sosdata: {}".format(type(sosdata)))
    print("len sosdata: {}".format(len(sosdata)))
    for impath, sos, lblids, label in zip(*sosdata):
        postsos[label].append([osp.join("http://hydra2.visenze.com:4567/", impath), sos])
        subcluster[label][int(lblids)].append([osp.join("http://hydra2.visenze.com:4567/", impath), lblids])

    for lbl in postsos:
        postsos[lbl] = sorted(postsos[lbl], key=lambda x: x[1], reverse=True)[:20]

    # filter the subcluster, remove one less than 3 samples
    for lbl in subcluster:
        for sublbl in subcluster[lbl]:
            l = min(len(subcluster[lbl][sublbl]), 10)
            subcluster[lbl][sublbl] = subcluster[lbl][sublbl][:l]

    return report_fullpath, postsos, subcluster

def run_embedding_search(modelpath, dsname, impath):
    """
    Call an external program which runs the Embedding Search and return the
    json file results. The function gets the json file and parse to
    corresponding format.
    Parameters:
        :param modelpath: (str) path to the model
        :param dsname: (str) name of dataset
        :param impath: (str) path to the query image
    """
    fname = osp.splitext(osp.basename(impath))[0]
    output_path = "{}_{}.json".format(fname, dsname)
    output_dir = osp.join(embdir, "{}_{}_dir".format(fname, dsname))

    if not osp.exists(output_path):
        # command example
        # python extract_fpn_embedding.py -I lighting.jpg \
        # --model-dir model_final_unopt/
        # --gpu 2
        # -O lighting_output
        # -D midlevel_v3.7_train
        run_command([
            "python",
            "/mnt/ssd_01/khoa/python_scripts/detectron_tools/extract_fpn_embedding.py",
            "-I", impath,
            "--model-dir", modelpath,
            "--gpu", str(2),
            "-O", output_dir,
            "-D", dsname
        ])

    # load the results
    raw = json.load(open(output_path, "r"))
    ret = []
    for r in raw:
        # print(r)
        qbox = r[0]
        newqbox = qbox.replace("/mnt/raid_04/ssd_01/khoa/aodet","")
        qret = r[1]
        newqret = [[q.replace("/mnt/raid_04/ssd_01/khoa/aodet",""),d] for (q,d) in qret]
        ret.append([newqbox, newqret])

    return ret
def get_viz_files(vizfiles, baseimname):
    """
    return following list:
        'imname': imname,
        'fpn': fpn name,
        'cls': predicted class
    """
    ret = list()
    ll = len(baseimname)
    for fname in vizfiles:
        code = fname[ll+1:].split("_", 1)
        ret.append({'imname': fname,
                   'fpn': code[0],
                   'cls': code[1].split(".")[0]})

    return ret


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

def find_pr_values(jsondata, cat):
    """
    Find Precision-Recall values of `cat` category in the json data
    :param: jsondata (Dict): json data structure
    :param: cat (str): category
    :return: precision (list of float)
    :return: recall (list of float)
    """
    # only get PR values
    data = jsondata['precision_recall_curve']
    catid = None
    for plugin in data['plugins']:

        if 'labels' not in plugin.keys():
            continue
        if (plugin['labels'][0] == cat) \
                or (cat == "average" and len(plugin['labels']) > 1):
            catid = plugin['id']
            break

    assert catid is not None, \
        "could not find axis id of {}".format(catid)

    # use catid to find core data id

    dataid = None
    if catid[-3:] == "pts":
        catid = catid[:-3]

    for line in data['axes'][0]['lines']:
        if line['id'] == catid:
            dataid = line['data']
            break

    assert dataid is not None, \
        "could not find dataid from {}".format(catid)

    precision = []
    recall = []
    prdata = data['data'][dataid]
    for pts in prdata:
        recall.append(pts[0])
        precision.append(pts[1])

    return {'precision': precision, 'recall': recall}
if __name__ == "__main__":
    app.run(host='0.0.0.0', port='3456')
