"""
Heuristic Analysis for Detection Evaluation.

Basically, if we want to add a new heuristic to the analysis system, just
derive from the Heuristic class and implement the __call__ method

"""

from .common import rel_delta

PCT_THRES = 0.05

class HeuristicOutput:
    """
    This is a enum type for Heuristic Output. Essentially it enumerates all
    possible outputs of all heuristics.
    """
    MISSING_GROUND_TRUTH = 0
    DIFFICULT_GROUND_TRUTH = 1
    CONFUSING_GROUNG_TRUTH = 2


class Heuristic(object):
    """
    Heristic class, it define the condition of certain heuristic. By using this
    approach, you can desing very simple condition or very complicated one. It
    is up to you. But remmember: the output of the __call__ method must be
    boolean type
    """
    def __init__(self, targets, output, lbl):
        assert targets is not None
        assert output is not None
        assert lbl is not None

        self.targets = targets
        self.output = output
        self.lbl = lbl

    def __call__(self):
        raise NotImplementedError("Please derive this method to implement")

    def output(self):
        raise NotImplementedError("Return the HeuristicOutput Type")


class MissingGroundTruth(Heuristic):
    """
    We consider the data is missing ground-truth boxes when: the precision is
    believable, i.e., at least 50% and we look at the gap between the concept
    precision and the target precision.

    Logs:
        0.1: It is better if we can get the precision on devset and then we
        decide whether it is MissingGroundTruth or not

    """
    PRECISION_THRES = 0.5

    def __init__(self, targets, output, lbl):
        super(Heuristic, self).__init__(targets, output, lbl)

        self.trgprec = targets[lbl]['precision']
        self.prdprec = output.concept_prec(lbl)

    def output(self):
        return HeuristicOutput.MISSING_GROUND_TRUTH

    def __call__(self):
        return self.predprec > self.PRECISION_THRES \
            and rel_delta(self.trgprec, self.predprec) < PCT_THRES

class DifficultGroundTruth(Heuristic):
    """
    Another cases when the *box recall* is relatively high but it is still
    coult not achieve the target, the reason maybe because of the difficult
    ground-truth. Again, it is better if we can get the result from the devset
    first.
    """
    BOXRECALL_THRES = 0.5

    def __init__(self, targets, output, lbl):
        super(Heuristic, self).__init__(targets, output, lbl)

        self.trgrecall = targets[lbl]['recall']
        self.boxrecall = output.box_recall(lbl)

    def output(self):
        return HeuristicOutput.DIFFICULT_GROUND_TRUTH

    def __call__(self):
        return self.boxrecall > self.BOXRECALL_THRES \
            and rel_delta(self.trgrecall, self.boxrecall) < PCT_THRES


class ConfusingGroundTruth(Heuristic):
    """

    """
    pass
