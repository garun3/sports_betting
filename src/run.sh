SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CONFIG_FILE=${SCRIPT_DIR}'/config.py'

TARGET_KEY='LEAGUE'
REPLACEMENT_VALUE=\'$1\'

sed -i '' "s/\($TARGET_KEY *= *\).*/\1$REPLACEMENT_VALUE/" $CONFIG_FILE

TARGET_KEY='REFRESH_TIME'
REPLACEMENT_VALUE=$2

sed -i '' "s/\($TARGET_KEY *= *\).*/\1$REPLACEMENT_VALUE/" $CONFIG_FILE

#pip install -r requirements.txt

python gather_data.py
python process.py
python train.py
#python predict.py