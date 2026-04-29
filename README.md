# CE8_DeepLearningProject
Mini project for the deep learning course

## How to run
This model is hardcoded to run using Librispeech, therefore, singular files cannot currently be input for the time being. However, the smallest subset (test-clean) is not too large and is the default for evaluation.

in train.py set hyperparameters at the top of the file in the CONFIG to be the same as the hyperparameters in the paper. These are currently set to evaluate "ckpts/model_2026-04-19_01-34_epoch-270.pth". More can be seen in configs.cfg. Make sure parameters at the bottom are the same as the following, making sure to set the mode to eval.
`main(
        mode="eval",
        validate_during_training=False,
        distributed=False,
        load_from_ckpt_path="ckpts/model_2026-04-19_01-34_epoch-270.pth"
    )`

The outputs will be text comparisons of the ground truth and model predictions using beam search. 
