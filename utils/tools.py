import os

def init_args(args):
    # The details for the following folders/files can be found in the annotation of the function 'preprocess_AVA' below
    args.modelSavePath    = os.path.join(args.savePath, 'model')
    args.scoreSavePath    = os.path.join(args.savePath, 'score.txt')
    args.trialPathAVA     = os.path.join(args.dataPathAVA, 'Single_Annotation_E_MUMMER_csv')
    args.audioPathAVA     = os.path.join(args.dataPathAVA, 'croped_E_MUMMER_audio_SR')
    args.visualPathAVA    = os.path.join(args.dataPathAVA, 'croped_face_region')
    args.trainTrialAVA    = os.path.join(args.trialPathAVA, 'train.csv')

    if args.evalDataType == 'val':
        args.evalTrialAVA = os.path.join(args.trialPathAVA, 'val.csv')
        args.evalOrig     = os.path.join(args.trialPathAVA, 'val_orig.csv')
        args.evalCsvSave  = os.path.join(args.savePath,     'val_res.csv')
    else:
        args.evalTrialAVA = os.path.join(args.trialPathAVA, 'test.csv')
        args.evalOrig     = os.path.join(args.trialPathAVA, 'test_orig.csv')
        args.evalCsvSave  = os.path.join(args.savePath,     'test_res.csv')
    
    os.makedirs(args.modelSavePath, exist_ok = True)
    os.makedirs(args.dataPathAVA, exist_ok = True)
    return args
