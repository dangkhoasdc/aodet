# -*- coding: utf-8 -*-
"""
Confusion matrix
"""

from .common import f1

class ConfusionMatrix(object):
    """
    General Confusion Matrix.
    Views:
        - Row: Groundtruths.
        - Column: Predictions.
    For the sake of simplicity, we regard this Confusion Matrix as "Concept
    Confusion Matrix". Later, when we introduct Detection Confusion Matrix, we
    have another type of matrix: Box Confusion Matrix. Since Box confusion
    matrix only occurs in the box detectio problem, I want to separate it
    and only use this one for general purposes.


    """
    def __init__(self, classes):
        self.classes = classes

        self.cf_mat = {gc: {pc: 0 for pc in classes} for gc in classes}

    def set_value(self, gtc, prdc, v):
        """
        set value of the confusion matrix
        Parameters:
            :param gtc: groundtruth class
            :param prdc: prediction class
            :param v: value
        """
        assert gtc in self.classes and prdc in self.classes
        self.cf_mat[gtc][prdc] = v


    def get_classes(self):
        return self.classes


    def get(self, gtc, prdc):
        """
        Return value of a cell in the matrix
        Parameters:
            :param gtc: ground-truth class name.
            :param prdc: prediction class name.
        """
        assert gtc in self.classes and prdc in self.classes, \
            "parameter is not in the class names"
        return self.cf_mat[gtc][prdc]

    def total(self):
        """ return total number of samples  """
        return sum([sum(self.cf_mat[g][p] for p in self.classes)
                   for g in self.classes])

    def ret(self, cls):
        """ return total returns of a class  """
        return sum([self.cf_mat[g][cls] for g in self.classes])

    def ngts(self, cls):
        """ return total groundtruths """
        return sum(self.cf_mat[cls].values())

    def tp(self, cls):
        """return true positves of a given class """
        return self.cf_mat[cls][cls]

    def tn(self, cls):
        """return true negatives of a given class """
        return self.total() - self.tp(cls) - self.fp(cls) - self.fn(cls)

    def fp(self, cls):
        """return false positives of a given class """
        return sum(self.cf_mat[g][cls] for g in self.classes if g != cls)

    def fn(self, cls):
        """return false negatives of a given class """
        return sum(self.cf_mat[cls][p] for p in self.classes if p != cls)

    def fallout(self, cls):
        """ compute fallout
        fallout =   fp/(fp+tn)
        """
        return float(self.fp(cls))/(self.fp(cls) + self.tn(cls))

    def __len__(self):
        return len(self.classes)

    def prec(self, cls):
        return float(self.tp(cls))/self.ret(cls)


    def recall(self, cls):
        return float(self.tp(cls))/self.ngts(cls)

    def f1(self, cls):
        p = self.prec(cls)
        r = self.recall(cls)

        return f1(p, r)

    def fbeta(self, cls, beta):
        p = self.prec(cls)
        r = self.recall(cls)

        return (1 + beta**2.0) * p * r/ ((beta**2 * p) + r)

    def roc_point(self, cls):
        """
        compute ROC point to plot the graph
        Parameters:
            :param cls: class cls
        """
        return (self.fallout(cls), self.recall(cls))

    def pr_point(self, cls):
        return (self.recall(cls), self.prec(cls))



class DetectionConfusionMatrix(ConfusionMatrix):
    """
    This particular class handle the results from the Detectio Evaluation
    System, not the search system. Hence, we only handle two more columns:
        1. *NO_MATCH*
        2. *DUPLICATES*
    However, we should handle the standard format. Therefore, I used the one
    applied in the Tensorflow API.

    1. For each detection record, the algorithm extracts from the input file
    the ground-truth boxes and classes, along with the detected boxes, classes,
    and scores.

    2. Only detections with a score greater or equal than conf threshold are
    considered.  Anything that’s under this value is discarded.

    3. For each ground-truth box, the algorithm generates the IoU (Intersection
    over Union) with every detected box. A match is found if both boxes have an
    IoU greater or equal than IoU threshold.

    4. The list of matches is pruned to remove duplicates (ground-truth boxes
    that match with more than one detection box or vice versa). If there are
    duplicates, the best match (greater IoU) is always selected.

    5. The confusion matrix is updated to reflect the resulting matches between
    ground-truth and detections.

    6. Objects that are part of the ground-truth but weren’t detected are
    counted in the last column of the matrix (in the row corresponding to the
    ground-truth class) (*NO_MATCH*). Objects that were detected but aren’t
    part of the confusion matrix are counted in the last row of the matrix (in
    the column corresponding to the detected class) (*NO_GROUNDTRUTH_SPEC*).
    """
    def __init__(self, classes):
        super().__init__(classes)

        self.no_matches = {cls: 0 for cls in self.classes}
        self.duplicates = {cls: 0 for cls in self.classes}
        self.nogt_spec = {cls: 0 for cls in self.classes}


    def ngts(self, cls):
        """ return total groundtruths """
        return sum(list(self.cf_mat[cls].values())
                   + [self.no_matches[cls], self.duplicates[cls]])

    def total(self):
        """ return total number of samples  """
        return sum([sum(self.cf_mat[g][p] for p in self.classes)
                   for g in self.classes]
                   + list(self.no_matches.values())
                   + list(self.duplicates.values())
                   + list(self.nogt_spec.values()))

    def ret(self, cls):
        """ return total returns of a class  """
        return sum([self.cf_mat[g][cls] for g in self.classes]
                   + [self.nogt_spec[cls]])

    def concept_roc_point(self, cls):
        return self.roc_point(cls)

    def concept_pr_point(self, cls):
        return self.pr_point(cls)


    def set_nomatch(self, cls, v):
        """
        set no matches
        Parameters:
            :param cls: class name
            :param v: value
        """
        self.no_matches[cls] = v


    def set_duplicate(self, cls, v):
        self.duplicates[cls] = v


    def set_nogt_spec(self, cls, v):
        self.nogt_spec[cls] = v

    def get_nogt_spec(self, cls):
        return self.nogt_spec[cls]

    def get_nomatch(self, cls):
        return self.no_matches[cls]

    def get_duplicate(self, cls):
        return self.duplicates[cls]



    # box metrics

    def box_tp_prd(self, cls):
        """
        return number of boxes that return as `cls` regardless the label.
        Basically, it is the sum of the column `cls` in the original confusion
        matrix
        """
        return sum(self.cf_mat[l][cls] for l in self.get_classes())

    def box_tp_gt(self, cls):
        """
        box true positive of groundtruth is sum of all boxes detected regardless the
        class output. In other words, we only consider if the box is detected

        """
        return sum(list(self.cf_mat[cls].values()))

    def box_recall(self, cls):
        """
        box recall is:
            box_tp_gt/ngts
        """
        return float(self.box_tp_gt(cls))/self.ngts(cls)

    def num_detected(self, cls):
        return self.box_tp_pred(cls) + self.nogt_spec[cls]

    def box_prec(self, cls):
        """
        box precision:
            box_tp_pred/num_detected
        """
        return float(self.box_tp_pred(cls))/self.num_detected(cls)

    # concept matrics
    def concept_recall(self, cls):
        """ the concept recall actually is the *true recall *
        Parameters:
            :param cls: cls of class
        """
        return self.recall(cls)


    def concept_prec(self, cls):
        """ the concept precision actually is the *true precision *
        Parameters:
            :param cls: cls of class
        """
        return self.prec(cls)

    def concept_f1(self, cls):
        return self.f1(cls)


    def __str__(self):
        skeys = sorted(list(self.cf_mat.keys()))
        rj = max(list(map(len, skeys)))
        txt = " " * (rj+1) + " ".join(skeys) + "\n"
        for lbl in skeys:
            txt += lbl.ljust(rj) + " "
            txt += " ".join(list(map(lambda x: str(x[0]).rjust(len(x[1])),
                                     [[self.cf_mat[lbl][k], k] for k in skeys])))
            txt += "\n"

        return txt



