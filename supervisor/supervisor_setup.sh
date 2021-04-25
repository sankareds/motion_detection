#/usr/bin/easy_install supervisor==3.1.1
yes | cp supervisord /etc/systemd/system/supervisord.service
systemctl enable supervisord
#/sbin/chkconfig --add supervisord
#sed -i "s/<<GENESIS_CONFIG>>/${GENESIS_CONFIG}/g" supervisord.conf
#sed -i "s/<<ENV>>/${ENV}/g" supervisord.conf
yes | cp supervisord.conf /etc/supervisord.conf
systemctl daemon-reload
mkdir -p /var/run/supervisord
chmod 0700 /var/run/supervisord
mkdir -p /var/log/supervisord
chmod 0700 /var/log/supervisord
#yes | cp restart_supervisor /etc/cron.hourly/restart_supervisor
#chmod 0400 /etc/cron.hourly/restart_supervisor
mkdir -p /etc/supervisord/
yes | cp motion_detection.ini /etc/supervisor/
chmod 0700 /etc/supervisor/motion_detection.ini
