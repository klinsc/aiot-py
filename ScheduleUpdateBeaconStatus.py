import time
from datetime import datetime, timedelta
import sqlite3


def UpdateStatus(table, status):
    ''' Update status table '''
    with sqlite3.connect('home.db') as conn:
        c = conn.cursor()
        c.execute(f'''UPDATE '{table}' SET status = '{status}' WHERE id = 1''')


def ScheduleUpdateBeaconStatus():
    ''' Update beacon status every 10 seconds, if timestamp is more than 1 mins '''
    while True:
        time.sleep(1)
        with sqlite3.connect('home.db') as conn:
            c = conn.cursor()
            c.execute('''SELECT date FROM timestamp WHERE id = 1''')
            date = c.fetchone()
            conn.commit()

            # Convert date to datetime
            oldTime = datetime.strptime(date[0], '%Y-%m-%d %H:%M:%S')

            # Get current time
            currentTime = datetime.now()

            # Get difference between current time and old time
            diff = currentTime - oldTime

            # If difference is more than 1 mins, update status to 'away'
            if diff > timedelta(minutes=1):
                UpdateStatus('beacon', 'AWAY')


ScheduleUpdateBeaconStatus()
