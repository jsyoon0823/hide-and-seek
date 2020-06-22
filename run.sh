set -e
set -x

virtualenv -p python3
source ./bin/activate

chmod +x data
chmod +x hider
chmod +x master_only
chmod +x metrics
chmod +x seeker
chmod +x tmp

pip install tensorflow
pip install -r requirements.txt
python -m main_hide-and-seek