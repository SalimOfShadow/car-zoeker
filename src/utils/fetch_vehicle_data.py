import json
import requests



def fetch_vehicle_data(license_plate: str):
    response = requests.get(f"https://opendata.rdw.nl/resource/m9d7-ebf2.json?kenteken={license_plate}")
    if response.status_code == 200:
        data = response.json()
        if data != []:
            return {
                "plate_number": data[0]["kenteken"],
                "vehicle_brand": data[0]["merk"],
                "vehicle_model": data[0]["handelsbenaming"],
                "vehicle_category": data[0]["europese_voertuigcategorie"],
                "registration_date": data[0]["datum_eerste_toelating_dt"]
            }
        print(f"Unfortunately, we couldn't find any entry for a vehicle with this plate number: {license_plate}.")
        return{}
    else:
        print("Encountered an error while fetching vehicle's data")
        print(response.status_code)


