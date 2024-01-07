usage="Usage: $0 [-m model_dir] [-v]"
REPLACE_MODEL_DIR="src/tiny/v0.1/training/image_classification/trained_models"
MODEL_NAME="pretrainedResnet_quant"
use_original_model=False

# read opts
while getopts 'm:v' opt
do
	case "${opt}" in
    m) model_dir=${OPTARG};;
    v) verbose=1;;
    ?) echo $usage
        exit 2;;
	esac
done

# check if model path specified
if [ $model_dir ]; then
    echo "Model path specified: '$model_dir'"
    use_original_model=False
else
    echo "Model path not specified, using default model."
    use_original_model=True
fi


# check if model exists
if [ ! -f $model_dir ]; then
    echo "Model not found!"
    exit 1
fi

# delete old model in src && build dir
echo "Deleting old model in src and build dir."
rm "$REPLACE_MODEL_DIR/$MODEL_NAME.tflite" 2>/dev/null
rm "$REPLACE_MODEL_DIR/$MODEL_NAME.h" 2>/dev/null
rm "build/$REPLACE_MODEL_DIR/$MODEL_NAME.h" 2>/dev/null

# if model path specified: copy model to current dir, and rename it 
if [ $use_original_model = True ]; then
    echo "Using default model."
else
    echo "Using model in '$model_dir'"
    echo "Copying model to '$REPLACE_MODEL_DIR'"
    mkdir -p $REPLACE_MODEL_DIR
    cp $model_dir "$REPLACE_MODEL_DIR/$MODEL_NAME.tflite"
fi

# build (make prog)
echo "Building... (First time may take a while)"
if [ $verbose ]; then
    make prog
else
    make prog 1>/dev/null  2>/dev/null
fi


# check if build success
if [ $? -eq 0 ]; then
    echo "Build success!"
else
    echo "Build failed!"
    exit 1
fi

# load first time, gurarantee to have error
echo "First load, generate $MODEL_NAME.h file."
if [ $verbose ]; then
    make load
else
    make load 1>/dev/null 2>/dev/null
fi

# modify build/$REPLACE_MODEL_DIR/$MODEL_NAME.h
# add const to pretrainedResnet_quant_len variable.
echo "Modifying pretrainedResnet_quant.h..."
sed -i 's/unsigned int pretrainedResnet_quant_len =/unsigned const int pretrainedResnet_quant_len =/' "build/$REPLACE_MODEL_DIR/$MODEL_NAME.h"

# load again
echo "Second time Loading..."
echo "Press enter and type 'reboot' to reboot the board."
echo "Ctrl+C to exit after rebooted."
if [ $verbose ]; then
    make load
else
    make load 2>/dev/null
fi

