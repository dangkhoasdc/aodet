"""
Confusion matrix
"""

from .common import f1

class ConfusionMatrix(object):
    """

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
        return self.cf_mat[gtc][prdc]

    def total(self):
        """ return total number of samples  """
        return sum([sum(self.cf_mat[g][p] for p in self.classes)
                   for g in self.classes])

    def ret(self, name):
        """ return total returns of a class  """
        return sum([self.cf_mat[g][name] for g in self.classes])

    def ngts(self, name):
        """ return total groundtruths """
        return sum(self.cf_mat[name].values())

    def tp(self, name):
        """return true positves of a given class """
        return self.cf_mat[name][name]

    def tn(self, name):
        """return true negatives of a given class """
        return self.total() - self.tp(name) - self.fp(name) - self.fn(name)

    def fp(self, name):
        """return false positives of a given class """
        return sum(self.cf_mat[g][name] for g in self.classes if g != name)

    def fn(self, name):
        """return false negatives of a given class """
        return sum(self.cf_mat[name][p] for p in self.classes if p != name)

    def fallout(self, name):
        """ compute fallout
        fallout =   fp/(fp+tn)
        """
        return float(self.fp(name))/(self.fp(name) + self.tn(name))

    def __len__(self):
        return len(self.classes)

    def prec(self, name):
        return float(self.tp(name))/self.ret(name)


    def recall(self, name):
        return float(self.tp(name))/self.ngts(name)

    def f1(self, name):
        p = self.prec(name)
        r = self.recall(name)

        return f1(p, r)

    def fbeta(self, name, beta):
        p = self.prec(name)
        r = self.recall(name)

        return (1 + beta**2.0) * p * r/ ((beta**2 * p) + r)

    def roc_point(self, name):
        """
        compute ROC point to plot the graph
        Parameters:
            :param name: class name
        """
        return (self.fallout(name), self.recall(name))

    def pr_point(self, name):
        return (self.recall(name), self.prec(name))



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

    def ret(self, name):
        """ return total returns of a class  """
        return sum([self.cf_mat[g][name] for g in self.classes]
                   + [self.nogt_spec[name]])

    def concept_roc_point(self, name):
        return self.roc_point(name)

    def concept_pr_point(self, name):
        return self.pr_point(name)

    def box_recall(self, name):
        """
        compute box recall
        """
        pass

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


    def box_prec(self, name):
        pass

    def concept_recall(self, name):
        """ the concept recall actually is the *true recall *
        Parameters:
            :param name: name of class
        """
        return self.recall(name)


    def concept_prec(self, name):
        """ the concept precision actually is the *true precision *
        Parameters:
            :param name: name of class
        """
        return self.prec(name)

    def concept_f1(self, name):
        return self.f1(name)


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



