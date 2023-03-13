import requests

# Payload
sample_dict = {
                "workclass": "state_gov",
                "education": "11th",
                "marital_status": "Never_married",
                "occupation": "adm_clerical",
                "relationship": "not_in_family",
                "race": "white",
                "sex": "Male",
                "native_country": "Cuba",
                "age": 29,
                "fnlwgt": 77516,
                "education_num": 13,
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40
            }
                  
url = "https://pedro-demo-heroku.herokuapp.com/inference"
post_response = requests.post(url, json=sample_dict)
print(post_response.status_code)
print(post_response.content)
