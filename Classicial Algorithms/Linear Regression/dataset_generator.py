import csv
import random

base_price = 100000
price_per_sqm = 5000


def generate_random():
    size = random.uniform(70, 500)
    noise = random.uniform(-200000, 200000)

    p = random.randint(0, 1)
    if p <= 0.3:
        noise += 300000
    price = price_per_sqm * size + base_price + noise

    return size, price


pairs = [generate_random() for _ in range(200)]

with open('dataset.csv', 'w', newline='') as csvfile:
    fieldnames = ['size_square_meter', 'price']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for pair in pairs:
        writer.writerow({'size_square_meter': pair[0], 'price': pair[1]})

print("CSV file created")
