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

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

pip install gunicorn
gunicorn -w 4 -b 127.0.0.1:5295 app:app --timeout 600
