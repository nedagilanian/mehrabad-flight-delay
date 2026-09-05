import pandas as pd
import numpy as np
import random

def generate_flight_data(n=500):
    airlines = ['IranAir', 'Mahan', 'Qeshm', 'Caspian', 'Aseman']
    destinations = ['Mashhad', 'Shiraz', 'Tabriz', 'Isfahan', 'Ahvaz']
    days = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    data = []
    for i in range(n):

        airline = random.choice(airlines)
        destination = random.choice(destinations)
        weekday = random.choice(days)
        scheduled_hour = random.randint(0, 23)

        delay_minutes = 0

        # اثر ایرلاین
        if airline == "Caspian":
            delay_minutes += 20
        elif airline == "Aseman":
            delay_minutes += 15
        elif airline == "IranAir":
            delay_minutes += 5

        # اثر مقصد
        if destination == "Mashhad":
            delay_minutes += 10
        elif destination == "Ahvaz":
            delay_minutes += 8

        # ساعات شلوغ
        if 18 <= scheduled_hour <= 23:
            delay_minutes += 15
        elif 6 <= scheduled_hour <= 9:
            delay_minutes += 8

        # آخر هفته
        if weekday in ["Thursday", "Friday"]:
            delay_minutes += 10

        # نویز تصادفی
        delay_minutes += np.random.randint(-5, 10)

        delay_minutes = max(0, delay_minutes)

        delay_status = 1 if delay_minutes > 15 else 0

        data.append({
            'Airline': airline,
            'Destination': destination,
            'Weekday': weekday,
            'ScheduledHour': scheduled_hour,
            'DelayMinutes': delay_minutes,
            'Delayed': delay_status
        })
        
    return pd.DataFrame(data)

df = generate_flight_data()
df.to_csv("mehrabad_flights.csv", index=False)
print("✅ Dataset saved to mehrabad_flights.csv")
