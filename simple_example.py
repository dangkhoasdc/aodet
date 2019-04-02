"""
Print the analysis messages
"""

import argparse
import pandas as pd
import aodet.analysis as ana
import aodet.io as aio

# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze confusion matrix "
                                     " of the detection results")
    parser.add_argument('-I', '--input', required=True, type=str,
                        help="input csv file")
    parser.add_argument("-M", "--mapping", type=str, required=True, \
                        help="mapping file")
    parser.add_argument("-T", "--target", type=str, required=True, \
                        help="Target file")
    return parser.parse_args()


def main():
    args = parse_args()
    targets = pd.read_csv(args.target, index_col=0).to_dict('index')
    results = aio.Reader.from_detection_sys(args.input, args.mapping)

    print("Recall")
    for cls in results.get_classes():
        print("{}: {}".format(cls ,results.concept_recall(cls)))
        
        
    print("Precision")
    for cls in results.get_classes():
        print("{}: {}".format(cls ,results.concept_prec(cls)))

    detection_analysis = ana.TargetAnalysis(targets, results)
    report = detection_analysis.analyze()
    
    print("Classes achieved target: ")
    for lbl in results.get_classes():
        if report[lbl] == ana.ResultType.ACHIEVED:
            print(lbl)
    print("===========================")
    
    
    print("Classes underperforms : ")
    for lbl in results.get_classes():
        if report[lbl] == ana.ResultType.UNDERPERFORMED:
            print(lbl)
    print("===========================")
    
    
    print("Classes have precision/recall unbalanced")
    for lbl in results.get_classes():
        if report[lbl] == ana.ResultType.CONSIDERING:
            print(lbl)
    print("===========================")

if __name__=="__main__":
    main()
