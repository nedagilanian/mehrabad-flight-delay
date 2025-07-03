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
        delay_minutes = max(0, int(np.random.normal(loc=10, scale=20)))  # میانگین ۱۰ دقیقه تأخیر
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
