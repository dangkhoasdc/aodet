"""
Analysis Confusion Matrix and Detection Result
"""

from .common import comp_list

class Suggestion:
    HIGH_FALSE_POSITIVES = [
        'Missing ground truth',
        'Low threshold',
        'Model bias'
    ]

    LOW_RECALL = [
        'Difficult groundtruth',
        'Wrong annotations and guidelines',
    ]

class ResultType:
    ACHIEVED = 0
    CONSIDERING = 1
    UNDERPERFORMED = 2

class TargetAnalysis(object):
    def __init__(self, targets=None, model_results=None):
        self.set_targets(targets)
        self.set_model_results(model_results)
        self.called_analyze = False


    def set_targets(self, targets):
        self.targets = targets


    def set_model_results(self, model_results):
        self.model_results = model_results

    def get_high_fp(self):
        """ return classes with high false positives and its values """
        pass

    def get_finetuning_thresholds(self):
        """ return list of classes which can be solved by finetuning
        threshold.
        Parameters:
            :return: list of classes (str)
        """
        if not self.called_analyze:
            self.analyze()

        for cls, ret in self.model_results:
            if self.report[cls] != ResultType.CONSIDERING:
                continue


        return None


    def analyze(self):
        """
        conduct analysis by comparing with the targets recall and precision
        """
        assert self.targets is not None
        assert self.model_results is not None
        print(self.targets.keys())
        print(self.model_results.get_classes())
        assert comp_list(list(self.targets.keys()),
                         self.model_results.get_classes()),\
            "wrong class info between targets and results"

        self.report = {cls: None for cls in self.targets.keys()}

        for cls in self.model_results.get_classes():
            trg_rec = self.targets[cls]['recall']
            trg_prec = self.targets[cls]['precision']

            mrec = self.model_results.concept_recall(cls)
            mprec = self.model_results.concept_prec(cls)

            if mprec >=trg_prec and mrec >= trg_rec:
                self.report[cls] = ResultType.ACHIEVED
            elif mprec <trg_prec and mrec < trg_rec:
                self.report[cls] = ResultType.UNDERPERFORMED
            else:
                self.report[cls] = ResultType.CONSIDERING
        self.called_analyze = True
        return self.report
