class config():

    MAX_FULLTEXT_LENGTH = 768
    MAX_SUMMARY_LENGTH = 200
    NUM_CLASSIFIERS = 23
    TRAIN_EVAL_SPLIT_RATIO = 0.75
    K_FOLDS = 3
    
    # actions list
    ACTION = {}
    ACTION['Distribute'] = 0
    ACTION['Modify'] = 1
    ACTION['Commercial Use'] = 2
    ACTION['Hold Liable'] = 3
    ACTION['Include Copyright'] = 4
    ACTION['Include License'] = 5
    ACTION['Sublicense'] = 6
    ACTION['Use Trademark'] = 7
    ACTION['Private Use'] = 8
    ACTION['Disclose Source'] = 9
    ACTION['State Changes'] = 10
    ACTION['Place Warranty'] = 11
    ACTION['Include Notice'] = 12
    ACTION['Include Original'] = 13
    ACTION['Give Credit'] = 14
    ACTION['Use Patent Claims'] = 15
    ACTION['Rename'] = 16
    ACTION['Relicense'] = 17
    ACTION['Contact Author'] = 18
    ACTION['Include Install Instructions'] = 19
    ACTION['Compensate for Damages'] = 20
    ACTION['Statically Link'] = 21
    ACTION['Pay Above Use Threshold'] = 22

    # map names to labels
    NAME2LABEL={}
    NAME2LABEL['none'] = 0
    NAME2LABEL['can'] = 1
    NAME2LABEL['cannot'] = 2
    NAME2LABEL['must'] = 3
    
    LABEL2ACTION = {}
    LABEL2ACTION[0] = 'Distribute'
    LABEL2ACTION[1] = 'Modify'
    LABEL2ACTION[2] = 'Commercial Use'
    LABEL2ACTION[3] = 'Hold Liable'
    LABEL2ACTION[4] = 'Include Copyright'
    LABEL2ACTION[5] = 'Include License'
    LABEL2ACTION[6] = 'Sublicense'
    LABEL2ACTION[7] = 'Use Trademark'
    LABEL2ACTION[8] = 'Private Use'
    LABEL2ACTION[9] = 'Disclose Source'
    LABEL2ACTION[10] = 'State Changes'
    LABEL2ACTION[11] = 'Place Warranty'
    LABEL2ACTION[12] = 'Include Notice'
    LABEL2ACTION[13] = "Include Original"
    LABEL2ACTION[14] = 'Give Credit'
    LABEL2ACTION[15] = 'Use Patent Claims'
    LABEL2ACTION[16] = 'Rename'
    LABEL2ACTION[17] = 'Relicense'
    LABEL2ACTION[18] = 'Contact Author'
    LABEL2ACTION[19] = 'Include Install Instructions'
    LABEL2ACTION[20] = 'Compensate for Damages'
    LABEL2ACTION[21] = 'Statically Link'
    LABEL2ACTION[22] = 'Pay Above Use Threshold'