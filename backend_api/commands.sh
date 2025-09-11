sudo apt install git
sudo apt install -y python3 python3-pip python3-venv

git config --global user.name "rifkymuth"
git config --global user.email "rifkyvirtualacc@gmail.com"
git clone https://github.com/rifkymuth/ta2_userprofiling.git

cd ta2_userprofiling/backend_api
gcloud storage cp --recursive  gs://user_profiling_ta2/ .


sudo apt install nginx
sudo nano /etc/nginx/sites-available/flaskapp

sudo ln -s /etc/nginx/sites-available/flaskapp /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
sudo nginx -s reload

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

pip install gunicorn
gunicorn -w 4 -b 127.0.0.1:5295 app:app --timeout 600 --daemon
gunicorn -w 4 -b 127.0.0.1:5295 app:app --timeout 600 --daemon --pid gunicorn.pid
# kill gunicorn
ps aux | grep gunicorn
killall -9 gunicorn
kill -TERM $(cat gunicorn.pid)
sudo nano .env


sftp root@31.97.221.152
put  model_word2vec.model /root/user_profiling_ta2/models/
put  slang_id.txt /root/user_profiling_ta2/models/
put  stopwords_id.txt /root/user_profiling_ta2/models/
put -r my_distilbert_sentimen /root/user_profiling_ta2/models/
put -r my_indobert_topic_classification /root/user_profiling_ta2/models/
