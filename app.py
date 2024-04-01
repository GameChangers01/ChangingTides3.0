from flask import Flask, render_template, url_for, request
import numpy as np
# import tensorflow as tf
# import os
# from sklearn.ensemble import (
#     GradientBoostingRegressor,
#     RandomForestRegressor,
#     AdaBoostRegressor,
#     VotingRegressor,
# )
# from sklearn.svm import SVR
# from sklearn.svm import SVR
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import (
#     GradientBoostingRegressor,
#     RandomForestRegressor,
#     VotingRegressor,
# )
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.experimental import enable_hist_gradient_boosting
# from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import make_pipeline
# from sklearn.ensemble import VotingRegressor
# from xgboost import XGBRegressor
# from sklearn.svm import SVR
# from tensorflow import keras
# from tensorflow.keras import layers
# from sklearn.svm import SVR
# from sklearn.ensemble import StackingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.neural_network import MLPRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
import requests
import smtplib


app = Flask(__name__, static_url_path="/static")


# index page
@app.route("/")
def index():
    return render_template("index.html")


# SignUp page
@app.route("/signup")
def signup():
    return render_template("Signup.html")


# Login page
@app.route("/login")
def login():
    return render_template("Login.html")


@app.route("/detect")
def detect():
    import requests
    from bs4 import BeautifulSoup

    city = "hyderabad"
    url = "https://www.google.com/search?q=" + "weather" + city
    html = requests.get(url).content
    soup = BeautifulSoup(html, "html.parser")
    temp = soup.find("div", attrs={"class": "BNeawe iBp4i AP7Wnd"}).text
    str = soup.find("div", attrs={"class": "BNeawe tAd8D AP7Wnd"}).text
    print(str)
    data = str.split("\n")
    time = data[0]
    sky = data[1]
    listdiv = soup.findAll("div", attrs={"class": "BNeawe s3v9rd AP7Wnd"})
    strd = listdiv[5].text
    pos = strd.find("Wind")
    other_data = strd[pos:]
    print("Temperature is", temp)
    print("Time: ", time)
    print("Sky Description: ", sky)
    print(other_data)
    weather = time + " " + temp + " " + sky
    return render_template("detect.html", time=time, temp=temp, sky=sky)


@app.route("/contactus")
def contactus():
    return render_template("contactus.html")


def mail_send(user_email, city, year, max_t, min_t, rain, aqi, sealevel, ozone):

    email = "climatechangeanalysisvbb@gmail.com"
    receiver_email = user_email
    subject = "Weather Report for " + city
    template = (
        """
    <!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">

    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="x-apple-disable-message-reformatting">
        <meta name="format-detection" content="telephone=no,address=no,email=no,date=no,url=no">

        <meta name="color-scheme" content="light">
        <meta name="supported-color-schemes" content="light">

        
        <!--[if !mso]><!-->
          
          <link rel="preload" as="style" href="https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&display=swap">
          <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&display=swap">

          <style type="text/css">
          // TODO: fix me!
            @import url(https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&family=Open+Sans:ital,wght@0,400;0,700;1,400;1,700&display=swap);
        </style>
        
        <!--<![endif]-->

        <!--[if mso]>
          <style>
              // TODO: fix me!
              * {
                  font-family: sans-serif !important;
              }
          </style>
        <![endif]-->
    
        
        <!-- NOTE: the title is processed in the backend during the campaign dispatch -->
        <title></title>

        <!--[if gte mso 9]>
        <xml>
            <o:OfficeDocumentSettings>
                <o:AllowPNG/>
                <o:PixelsPerInch>96</o:PixelsPerInch>
            </o:OfficeDocumentSettings>
        </xml>
        <![endif]-->
        
    <style>
        :root {
            color-scheme: light;
            supported-color-schemes: light;
        }

        html,
        body {
            margin: 0 auto !important;
            padding: 0 !important;
            height: 100% !important;
            width: 100% !important;

            overflow-wrap: break-word;
            -ms-word-break: break-all;
            -ms-word-break: break-word;
            word-break: break-all;
            word-break: break-word;
        }


        
  direction: undefined;
  center,
  #body_table {
    
  }

  ul, ol {
    padding: 0;
    margin-top: 0;
    margin-bottom: 0;
  }

  li {
    margin-bottom: 0;
  }

  

  .list-block-list-outside-left li {
    margin-left: 20px !important;
  }

  .list-block-list-outside-right li {
    margin-right: 20px !important;
  }

  
    .paragraph {
      font-size: 15px;
      font-family: Open Sans, sans-serif;
      font-weight: normal;
      font-style: normal;
      text-align: start;
      line-height: 1;
      text-decoration: none;
      color: #5f5f5f;
      
    }
  

    .heading1 {
      font-size: 32px;
      font-family: Open Sans, sans-serif;
      font-weight: normal;
      font-style: normal;
      text-align: start;
      line-height: 1;
      text-decoration: none;
      color: #000000;
      
    }
  

    .heading2 {
      font-size: 26px;
      font-family: Open Sans, sans-serif;
      font-weight: normal;
      font-style: normal;
      text-align: start;
      line-height: 1;
      text-decoration: none;
      color: #000000;
      
    }
  

    .heading3 {
      font-size: 19px;
      font-family: Open Sans, sans-serif;
      font-weight: normal;
      font-style: normal;
      text-align: start;
      line-height: 1;
      text-decoration: none;
      color: #000000;
      
    }
  

    .list {
      font-size: 15px;
      font-family: Open Sans, sans-serif;
      font-weight: normal;
      font-style: normal;
      text-align: start;
      line-height: 1;
      text-decoration: none;
      color: #5f5f5f;
      
    }
  

  p a, 
  li a {
    
  display: inline-block;  
    color: #5457FF;
    text-decoration: none;
    font-style: normal;
    font-weight: normal;

  }

  .button-table a {
    text-decoration: none;
    font-style: normal;
    font-weight: normal;
  }

  .paragraph > span {text-decoration: none;}.heading1 > span {text-decoration: none;}.heading2 > span {text-decoration: none;}.heading3 > span {text-decoration: none;}.list > span {text-decoration: none;}


        * {
            -ms-text-size-adjust: 100%;
            -webkit-text-size-adjust: 100%;
        }

        div[style*="margin: 16px 0"] {
            margin: 0 !important;
        }

        #MessageViewBody,
        #MessageWebViewDiv {
            width: 100% !important;
        }

        table {
            border-collapse: collapse;
            border-spacing: 0;
            mso-table-lspace: 0pt !important;
            mso-table-rspace: 0pt !important;
        }
        table:not(.button-table) {
            border-spacing: 0 !important;
            border-collapse: collapse !important;
            table-layout: fixed !important;
            margin: 0 auto !important;
        }

        th {
            font-weight: normal;
        }

        tr td p {
            margin: 0;
        }

        img {
            -ms-interpolation-mode: bicubic;
        }

        a[x-apple-data-detectors],

        .unstyle-auto-detected-links a,
        .aBn {
            border-bottom: 0 !important;
            cursor: default !important;
            color: inherit !important;
            text-decoration: none !important;
            font-size: inherit !important;
            font-family: inherit !important;
            font-weight: inherit !important;
            line-height: inherit !important;
        }

        .im {
            color: inherit !important;
        }

        .a6S {
            display: none !important;
            opacity: 0.01 !important;
        }

        img.g-img+div {
            display: none !important;
        }

        @media only screen and (min-device-width: 320px) and (max-device-width: 374px) {
            u~div .contentMainTable {
                min-width: 320px !important;
            }
        }

        @media only screen and (min-device-width: 375px) and (max-device-width: 413px) {
            u~div .contentMainTable {
                min-width: 375px !important;
            }
        }

        @media only screen and (min-device-width: 414px) {
            u~div .contentMainTable {
                min-width: 414px !important;
            }
        }
    </style>

    <style>
        @media only screen and (max-device-width: 640px) {
            .contentMainTable {
                width: 100% !important;
                margin: auto !important;
            }
            .single-column {
                width: 100% !important;
                margin: auto !important;
            }
            .multi-column {
                width: 100% !important;
                margin: auto !important;
            }
            .imageBlockWrapper {
                width: 100% !important;
                margin: auto !important;
            }
        }
        @media only screen and (max-width: 640px) {
            .contentMainTable {
                width: 100% !important;
                margin: auto !important;
            }
            .single-column {
                width: 100% !important;
                margin: auto !important;
            }
            .multi-column {
                width: 100% !important;
                margin: auto !important;
            }
            .imageBlockWrapper {
                width: 100% !important;
                margin: auto !important;
            }
        }
    </style>
    
    
<!--[if mso | IE]>
    <style>
        .list-block-outlook-outside-left {
            margin-left: -18px;
        }
    
        .list-block-outlook-outside-right {
            margin-right: -18px;
        }

        a:link, span.MsoHyperlink {
            mso-style-priority:99;
            
  display: inline-block;  
    color: #5457FF;
    text-decoration: none;
    font-style: normal;
    font-weight: normal;

        }
    </style>
<![endif]-->


    </head>

    <body width="100%" style="margin: 0; padding: 0 !important; mso-line-height-rule: exactly; background-color: #F5F6F8;">
        <center role="article" aria-roledescription="email" lang="en" style="width: 100%; background-color: #F5F6F8;">
            <!--[if mso | IE]>
            <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" id="body_table" style="background-color: #F5F6F8;">
            <tbody>    
                <tr>
                    <td>
                    <![endif]-->
                        <table align="center" role="presentation" cellspacing="0" cellpadding="0" border="0" width="640" style="margin: auto;" class="contentMainTable">
                            <tr class="wp-block-editor-spacerblock-v1"><td style="background-color:#F5F6F8;line-height:50px;font-size:50px;width:100%;min-width:100%">&nbsp;</td></tr><tr class="wp-block-editor-imageblock-v1"><td style="background-color:#295393;padding-top:32px;padding-bottom:32px;padding-left:32px;padding-right:32px" align="center"><table align="center" width="374.4" class="imageBlockWrapper" style="width:374.4px" role="presentation"><tbody><tr><td style="padding:0"><img src="https://api.smtprelay.co/userfile/de0963c7-5c81-4532-8c31-9e1e92ae35a5/logo2024-03-21T21_02_27.png" width="374.4" height="" alt="" style="border-radius:0px;display:block;height:auto;width:100%;max-width:100%;border:0" class="g-img"></td></tr></tbody></table></td></tr><tr class="wp-block-editor-imageblock-v1"><td style="background-color:#ffffff;padding-top:0;padding-bottom:0;padding-left:0;padding-right:0" align="center"><table align="center" width="640" class="imageBlockWrapper" style="width:640px" role="presentation"><tbody><tr><td style="padding:0"><img src="https://api.smtprelay.co/userfile/de0963c7-5c81-4532-8c31-9e1e92ae35a5/planets-solar-system.jpeg" width="640" height="" alt="" style="border-radius:0px;display:block;height:auto;width:100%;max-width:100%;border:0" class="g-img"></td></tr></tbody></table></td></tr><tr class="wp-block-editor-headingblock-v1"><td valign="top" style="background-color:#ffffff;display:block;padding-top:64px;padding-right:32px;padding-bottom:32px;padding-left:32px;text-align:center"><p style="font-family:Open Sans, sans-serif;text-align:center;line-height:36.80px;font-size:32px;background-color:#ffffff;color:#000000;margin:0;word-break:normal" class="heading1"><span style="font-weight: bold" class="bold">Changing Tides</span></p></td></tr><tr class="wp-block-editor-paragraphblock-v1"><td valign="top" style="padding:0px 32px 32px 32px;background-color:#ffffff"><p class="paragraph" style="font-family:Open Sans, sans-serif;text-align:center;line-height:30.00px;font-size:15px;margin:0;color:#5f5f5f;word-break:normal">Climate shapes our world, painting it with the colors of seasons, breathing life into the rhythm of nature.</p></td></tr><tr class="wp-block-editor-headingblock-v1"><td valign="top" style="background-color:#F9FAFB;display:block;padding-top:42px;padding-right:32px;padding-bottom:16px;padding-left:32px;text-align:center"><p style="font-family:Open Sans, sans-serif;text-align:center;line-height:19.00px;font-size:19px;background-color:#F9FAFB;color:#000000;margin:0;word-break:normal" class="heading3">
                            <h2>Weather Report for """
        + str(city)
        + """</h2>
                            <p>Max Temperature: """
        + str(max_t)
        + "°C"
        + """</p>
                            <p>Min Temperature: """
        + str(min_t)
        + "°C"
        + """</p>
                            <p>Rainfall: """
        + str(rain)
        + " mm"
        + """</p>
                            <p>AQI: """
        + str(aqi)
        + " μg/m3"
        + """</p>
                            <p>Sea Level: """
        + str(sealevel)
        + " M"
        + """</p>
                            <p>Ozone: """
        + str(ozone)
        + " KM"
        + """</p>
                            <p>Year: """
        + str(year)
        + """</p>
                            <p>City: """
        + str(city)
        + """ </p>
                            </p></td></tr><tr class="wp-block-editor-headingblock-v1"><td valign="top" style="background-color:#ffffff;display:block;padding-top:52px;padding-right:32px;padding-bottom:16px;padding-left:32px;text-align:center"><p style="font-family:Open Sans, sans-serif;text-align:center;line-height:29.90px;font-size:26px;background-color:#ffffff;color:#000000;margin:0;word-break:normal" class="heading2">Thank you for using our Application, Save the Nature and Save the Earth...</p></td></tr><tr class="wp-block-editor-spacerblock-v1"><td style="background-color:#ffffff;line-height:32px;font-size:32px;width:100%;min-width:100%">&nbsp;</td></tr><tr class="wp-block-editor-imageblock-v1"><td style="background-color:#ffffff;padding-top:0;padding-bottom:0;padding-left:0;padding-right:0" align="center"><table align="center" width="640" class="imageBlockWrapper" style="width:640px" role="presentation"><tbody><tr><td style="padding:0"><img src="https://api.smtprelay.co/userfile/de0963c7-5c81-4532-8c31-9e1e92ae35a5/panoramic-shot-from-drone-nature-moldova-sunset.jpeg" width="640" height="" alt="" style="border-radius:0px;display:block;height:auto;width:100%;max-width:100%;border:0" class="g-img"></td></tr></tbody></table></td></tr>
                        </table>
                    <!--[if mso | IE]>
                    </td>
                </tr>
            </tbody>
            </table>
            <![endif]-->
        </center>
    </body>
</html>
    """
    )
    message = f"Subject: {subject}\n"

    # Include the HTML content as part of the message
    message += "MIME-Version: 1.0\n"
    message += "Content-Type: text/html\n"
    message += template

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(email, "rhcutuankixeercv")
    server.sendmail(email, receiver_email, message.encode("utf8"))
    server.quit()


@app.route("/result", methods=["POST", "GET"])
def Result():
    if request.method == "POST":
        ans = ""
        max_t = 49
        min_t = -2
        avg_t = 30
        rain = 50
        aqi = 12
        co2 = 0.5
        sealevel = 5
        ozone = 10000
        city = request.form["city"]
        year = request.form["year"]
        user_email = request.cookies.get("userEmail")
        print(user_email)
        y1 = int(year)

        ans = city + year
        if city == "Bangalore":
            # aqi = predict_air("AQI_prediction_model.pkl", year)
            # sealevel = predict_sea("sealevel_stacking_model.pkl", year)
            aqi = air_prediction(y1)
            # co2 = co2_emissions(year)
            max_t, min_t = predict_temp(y1)
            sealevel = predict_sea(y1)
            rain = predict_total_rainfall_for_year(y1)
            ozone = predict_maximum_ozone(y1)
            mail_send(user_email, city, year, max_t, min_t, rain, aqi, sealevel, ozone)
            return render_template(
                "result.html",
                max_t=max_t,
                min_t=min_t,
                rain=rain,
                aqi=aqi,
                sealevel=sealevel,
                ozone=ozone,
                year=year,
                city=city,
                display_charts=True,
            )

        elif city == "Hyderabad":

            aqi = air_prediction_hyd(y1)
            max_t, min_t = predict_temp_hyd(y1)
            sealevel = predict_sea(y1)
            rain = predict_total_rainfall_for_year_hyd(y1)
            ozone = predict_maximum_ozone(y1)
            mail_send(user_email, city, year, max_t, min_t, rain, aqi, sealevel, ozone)
            return render_template(
                "result.html",
                max_t=max_t,
                min_t=min_t,
                rain=rain,
                aqi=aqi,
                sealevel=sealevel,
                ozone=ozone,
                year=year,
                city=city,
                display_charts=False,
            )


@app.route("/home")
def home():
    return render_template("home.html")


def predict_pm10_level(year):
    # Load your dataset
    df = pd.read_excel("data\Air Qualtity Index.xlsx")

    df = df.dropna(subset=["Year "])

    X = df[["Year ", " Estimated Avg PM10 Levels (μg/m3) "]]
    y = df[" Estimated Avg PM10 Levels (μg/m3) "]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVR(kernel="linear")
    model.fit(X_scaled, y)

    input_pm10_level = 0

    input_data_scaled = scaler.transform([[year, input_pm10_level]])
    predicted_pm10_level = model.predict(input_data_scaled)[0]

    return predicted_pm10_level


def air_prediction(year):
    prd = {
        2020: 196.77,
        2021: 197.82826009114584,
        2022: 198.3904979296875,
        2023: 198.95483137916664,
        2024: 199.521579921875,
        2025: 200.0912488606771,
        2026: 200.6630278026771,
        2027: 201.23692043614584,
        2028: 201.8125148925781,
        2029: 202.39027408854164,
        2030: 202.9689837239583,
        2031: 203.5491248372396,
        2032: 204.13086969401042,
        2033: 204.71320110677084,
        2034: 205.2978515625,
        2035: 205.88413704427084,
        2036: 206.4712551875,
        2037: 207.0586692809271,
        2038: 207.646385671875,
        2039: 208.234130859375,
        2040: 208.8211401494792,
        2041: 209.40815559895834,
        2042: 209.9950673424479,
        2043: 210.5817684778646,
        2044: 211.16815356445312,
        2045: 211.75411979166666,
        2046: 212.33950519791666,
        2047: 212.92421630859375,
        2048: 213.50850158691406,
        2049: 214.09296875,
        2050: 214.67578684505206,
        2051: 215.25859090972916,
        2052: 215.8410682678223,
        2053: 216.42378011067708,
        2054: 217.00650423177083,
        2055: 217.5896923828125,
        2056: 218.17456461615626,
        2057: 218.76020431510416,
        2058: 219.34646606445312,
        2059: 219.93316243489584,
        2060: 220.52075927734375,
        2061: 221.10918294270834,
        2062: 221.69827473958334,
        2063: 222.28817138671875,
        2064: 222.87869669596355,
        2065: 223.46942443880208,
        2066: 224.06008275390626,
        2067: 224.65060424804688,
        2068: 225.24075927734376,
        2069: 225.8302080485026,
        2070: 226.419338671875,
        2071: 227.00784016927084,
        2072: 227.5956268310547,
        2073: 228.18265787760416,
        2074: 228.76885426888022,
        2075: 229.35427968343098,
        2076: 229.93890686035156,
        2077: 230.52341796875,
        2078: 231.10769653320312,
        2079: 231.69185384114584,
        2080: 232.27588993326822,
        2081: 232.85951232910156,
        2082: 233.44302693684895,
        2083: 234.0263077122396,
        2084: 234.60933430989584,
        2085: 235.19205403645834,
        2086: 235.7743585205078,
        2087: 236.35619481336802,
        2088: 236.93757169596356,
        2089: 237.51875813802084,
        2090: 238.09997329752604,
        2091: 218.21907552083334,
        2092: 218.45633951822916,
        2093: 218.6936661702474,
        2094: 218.93104329427084,
        2095: 219.16847127278647,
        2096: 219.40595029622397,
        2097: 219.6434834798177,
        2098: 219.88107401529948,
        2099: 220.11871910024042,
    }

    return prd[year]


def predict_sea(year):
    sv = {
        2020: 204.5449895120464,
        2021: 204.58056879026174,
        2022: 204.61666775379217,
        2023: 204.65320209315445,
        2024: 204.69008866688569,
        2025: 204.7272457509498,
        2026: 204.76459327794595,
        2027: 204.80205306505906,
        2028: 204.83954902976728,
        2029: 204.8770073923984,
        2030: 204.91435686471345,
        2031: 204.95152882378756,
        2032: 204.9884574705539,
        2033: 205.02507997247807,
        2034: 205.06133658993377,
        2035: 205.09717078595787,
        2036: 205.1325293191702,
        2037: 205.1673623197538,
        2038: 205.2016233484972,
        2039: 205.23526943900936,
        2040: 205.26826112332117,
        2041: 205.30056244118913,
        2042: 205.33214093351427,
        2043: 205.36296762038046,
        2044: 205.3930169643042,
        2045: 205.4222668193671,
        2046: 205.4506983669756,
        2047: 205.4782960390575,
        2048: 205.50504742956429,
        2049: 205.5309431951942,
        2050: 205.55597694629506,
        2051: 205.5801451289359,
        2052: 205.6034468991594,
        2053: 205.62588399044364,
        2054: 205.64746057540538,
        2055: 205.66818312277525,
        2056: 205.6880602506664,
        2057: 205.7071025771374,
        2058: 205.72532256902628,
        2059: 205.7427343900016,
        2060: 205.75935374873467,
        2061: 205.7751977480583,
        2062: 205.7902847359236,
        2063: 205.80463415891802,
        2064: 205.81826641904786,
        2065: 205.83120273443217,
        2066: 205.84346500449016,
        2067: 205.85507568014526,
        2068: 205.8660576395016,
        2069: 205.87643406938707,
        2070: 205.88622835309206,
        2071: 205.8954639645716,
        2072: 205.90416436931673,
        2073: 205.91235293204193,
        2074: 205.9200528312793,
        2075: 205.9272869809153,
        2076: 205.93407795865647,
        2077: 205.9404479413614,
        2078: 205.94641864713486,
        2079: 205.95201128403755,
        2080: 205.95724650523113,
        2081: 205.96214437034405,
        2082: 205.96672431281854,
        2083: 205.97100511297162,
        2084: 205.97500487648597,
        2085: 205.97874101802847,
        2086: 205.98223024968223,
        2087: 205.9854885738692,
        2088: 205.98853128043373,
        2089: 205.99137294755627,
        2090: 205.9940274461659,
        2091: 205.9965079475234,
        2092: 205.99882693365294,
        2093: 206.0009962103063,
        2094: 206.0030269221561,
        2095: 206.00492956992184,
        2096: 206.0067140291494,
        2097: 206.0083895703748,
        2098: 206.00996488042068,
        2099: 206.0114480845873,
    }

    return sv[year]


def co2_emissions(year_to_predict):
    url = r"data\co-emissions-per-capita.csv"
    df = pd.read_csv(url)
    df_ind = df[df["Code"] == "IND"]
    features = ["Year"]
    target = "Annual"
    X = df_ind[features]
    y = df_ind[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    gb_model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    gb_model.fit(X_train, y_train)
    y_pred = gb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    year_to_predict = np.array([[year_to_predict]])
    predicted_emissions = gb_model.predict(year_to_predict)[0]
    return predicted_emissions


def predict_temp(year):

    predictions_max_dict = {
        2020: 44.51737841804966,
        2021: 44.937762802652195,
        2022: 45.35814718725474,
        2023: 45.77853157185727,
        2024: 46.19891595645983,
        2025: 46.61926789622144,
        2026: 47.03960905278543,
        2027: 47.45995020934944,
        2028: 47.880291365913436,
        2029: 48.30063252247744,
        2030: 48.720973679041435,
        2031: 49.14131483560543,
        2032: 49.56165599216943,
        2033: 49.98248686859897,
        2034: 50.40434480175467,
        2035: 50.82620273491036,
        2036: 51.24806066806607,
        2037: 51.669918601221774,
        2038: 52.09177653437748,
        2039: 52.51363446753319,
        2040: 52.93549240068887,
        2041: 53.35735033384458,
        2042: 53.779208267000286,
        2043: 54.20106620015599,
        2044: 54.6229241333117,
        2045: 55.044881484544405,
        2046: 55.46772932572216,
        2047: 55.89057716689993,
        2048: 56.3134250080777,
        2049: 56.73627284925547,
        2050: 57.15913015513432,
        2051: 57.581994444399626,
        2052: 58.00485873366493,
        2053: 58.42772302293024,
        2054: 58.85058731219553,
        2055: 59.27345160146085,
        2056: 59.69631589072614,
        2057: 60.11918017999146,
        2058: 60.542044469256766,
        2059: 60.96490875852206,
        2060: 61.38777304778736,
        2061: 61.81063733705267,
        2062: 62.233501626317974,
        2063: 62.65636591558328,
        2064: 63.079230204848585,
        2065: 63.50209449411389,
        2066: 63.924958783379196,
        2067: 64.3478230726445,
        2068: 64.77068736190981,
        2069: 65.19355165117511,
        2070: 65.61641594044042,
        2071: 66.03928022970572,
        2072: 66.46310896304404,
        2073: 66.88736474107199,
        2074: 67.31162051909996,
        2075: 67.73587629712793,
        2076: 68.16013207515591,
        2077: 68.58438785318387,
        2078: 69.00864363121184,
        2079: 69.43289940923981,
        2080: 69.8571551872678,
        2081: 70.28141096529575,
        2082: 70.70566674332372,
        2083: 71.12992252135169,
        2084: 71.55417829937964,
        2085: 71.97843407740763,
        2086: 72.4026898554356,
        2087: 72.82694563346357,
        2088: 73.25120141149152,
        2089: 73.6754571895195,
        2090: 74.09971296754748,
        2091: 74.52396874557544,
        2092: 74.9482245236034,
        2093: 75.37248030163138,
        2094: 75.79673607965935,
        2095: 76.22099185768732,
        2096: 76.64524763571528,
        2097: 77.06950341374325,
        2098: 77.49375919177123,
        2099: 77.9180149697992,
    }

    predictions_min_dict = {
        2020: 30.163015223775734,
        2021: 30.32567376917344,
        2022: 30.488332314571146,
        2023: 30.65099085996885,
        2024: 30.813652200742588,
        2025: 30.97631979572043,
        2026: 31.13898739069827,
        2027: 31.301654985676112,
        2028: 31.464322580653953,
        2029: 31.626990175631796,
        2030: 31.79073584463267,
        2031: 31.95541878939049,
        2032: 32.120087434932465,
        2033: 32.28475608047443,
        2034: 32.4494247260164,
        2035: 32.61409337155837,
        2036: 32.778762017100334,
        2037: 32.943430662642314,
        2038: 33.10809930818428,
        2039: 33.272767953726245,
        2040: 33.43743659926822,
        2041: 33.60210524481018,
        2042: 33.766773890352155,
        2043: 33.93144253589412,
        2044: 34.09611118143609,
        2045: 34.260779826978066,
        2046: 34.42544847252003,
        2047: 34.590118028227764,
        2048: 34.75479986244274,
        2049: 34.92051713036773,
        2050: 35.08624116172941,
        2051: 35.25196519309111,
        2052: 35.4176892244528,
        2053: 35.58341325581449,
        2054: 35.74913728717618,
        2055: 35.914861318537874,
        2056: 36.08055849254589,
        2057: 36.246232255099144,
        2058: 36.41190601765239,
        2059: 36.577579780205646,
        2060: 36.74325354275889,
        2061: 36.90892730531215,
        2062: 37.0746010678654,
        2063: 37.24027483041865,
        2064: 37.40594859297189,
        2065: 37.57162347916143,
        2066: 37.737299461262154,
        2067: 37.90297544336288,
        2068: 38.0686514254636,
        2069: 38.23432740756433,
        2070: 38.40000338966506,
        2071: 38.56567937176578,
        2072: 38.73135535386651,
        2073: 38.897031335967235,
        2074: 39.06270731806795,
        2075: 39.22838330016869,
        2076: 39.3940592822694,
        2077: 39.55973526437013,
        2078: 39.72541124647086,
        2079: 39.89108722857158,
        2080: 40.05676321067231,
        2081: 40.22243919277304,
        2082: 40.38811517487376,
        2083: 40.553791156974484,
        2084: 40.719467139075206,
        2085: 40.88514312117593,
        2086: 41.05081910327666,
        2087: 41.21649508537738,
        2088: 41.38217106747811,
        2089: 41.547847049578834,
        2090: 41.713523031679564,
        2091: 41.87919901378028,
        2092: 42.04487499588102,
        2093: 42.21055097798173,
        2094: 42.37622696008246,
        2095: 42.541902942183185,
        2096: 42.70757892428391,
        2097: 42.87325490638464,
        2098: 43.03893088848536,
        2099: 43.20460687058609,
    }
    return predictions_max_dict[year], predictions_min_dict[year]


def predict_sea_level(year):
    url = r"data\Sea level (1).xlsx"
    sea_level_df = pd.read_excel(url)
    features = ["Year "]
    target = "Level"
    if "Year " in sea_level_df.columns:
        X = sea_level_df[features]
        y = sea_level_df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        models = [
            (
                "gb",
                GradientBoostingRegressor(
                    n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42
                ),
            ),
            (
                "rf",
                RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42),
            ),
            (
                "xgb",
                XGBRegressor(
                    n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42
                ),
            ),
            ("svm", SVR(kernel="rbf")),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
                ),
            ),
            ("hist_gb", HistGradientBoostingRegressor(max_iter=500, random_state=42)),
        ]

        stacking_model = StackingRegressor(
            estimators=models, final_estimator=LinearRegression(), cv=5
        )

        stacking_model.fit(X_train, y_train)
        year_array = np.array([[year]])
        future_prediction = stacking_model.predict(
            pd.DataFrame({"Year ": year_array.flatten()})
        )[0]

        print(f"Year: {year}, Predicted Sea Level: {future_prediction}")

        return future_prediction


def predict_total_rainfall_for_year(year):
    rain = {
        2020: 78.34081137798493,
        2021: 76.59604070766854,
        2022: 108.21399934988919,
        2023: 66.37160618060669,
        2024: 74.01665891107635,
        2025: 84.17691415305943,
        2026: 66.80553569431893,
        2027: 54.57193848035758,
        2028: 83.88021583538857,
        2029: 97.1668202321891,
        2030: 65.85185111937759,
        2031: 91.13992969689883,
        2032: 45.36065778897888,
        2033: 61.231481663045265,
        2034: 79.62500630048594,
        2035: 111.997066582551,
        2036: 74.97850561115737,
        2037: 67.02714916261256,
        2038: 86.55155708635778,
        2039: 54.92518133961582,
        2040: 77.36003335080969,
        2041: 72.46903565292516,
        2042: 56.60878121484131,
        2043: 57.19502106347201,
        2044: 61.979378438300834,
        2045: 65.9953584778245,
        2046: 50.255966507499444,
        2047: 80.86570003819385,
        2048: 65.73315532340737,
        2049: 82.63925961574581,
        2050: 53.452742595310355,
        2051: 90.63913605227508,
        2052: 87.42844174066833,
        2053: 52.30789226386005,
        2054: 91.67815978064634,
        2055: 64.28832363532219,
        2056: 69.191814783659,
        2057: 66.55199688345384,
        2058: 74.55349518125604,
        2059: 77.42267279192079,
        2060: 71.85056706146638,
        2061: 68.9646800581301,
        2062: 103.34210192466996,
        2063: 83.95129084581167,
        2064: 48.96381776849543,
        2065: 86.6819570534597,
        2066: 75.75442511660809,
        2067: 83.3974000145053,
        2068: 89.04588699094586,
        2069: 67.11642316932678,
        2070: 75.44744305979567,
        2071: 67.66731650174445,
        2072: 100.0916935058923,
        2073: 72.75889639084143,
        2074: 66.32813467453926,
        2075: 93.2371479497415,
        2076: 63.78366799432948,
        2077: 106.97372179799079,
        2078: 80.79089403339366,
        2079: 74.15503903944843,
        2080: 73.49357728407499,
        2081: 87.90370051278478,
        2082: 86.37422506529329,
        2083: 99.09076858679884,
        2084: 57.54367425847493,
        2085: 100.34139266961313,
        2086: 63.27224561148967,
        2087: 72.49258776749393,
        2088: 99.69018040271715,
        2089: 85.37210906502479,
        2090: 78.87019738743324,
        2091: 71.49116931058543,
        2092: 79.16233450940639,
        2093: 79.34492049172887,
        2094: 88.06089257158358,
        2095: 62.940743520128024,
        2096: 83.61869378491191,
        2097: 69.12734018032373,
        2098: 102.43074726589242,
        2099: 103.9689999386332,
    }
    return rain[year]


def predict_maximum_ozone(year):
    ozone = {
        2020: 24150003.091906756,
        2021: 24580405.201538973,
        2022: 24646150.559034448,
        2023: 24711895.916529916,
        2024: 24777641.274025388,
        2025: 24843386.631520856,
        2026: 24909131.98901633,
        2027: 24974877.3465118,
        2028: 25040622.704007268,
        2029: 25106368.061502736,
        2030: 25172113.418998204,
        2031: 25237858.776493676,
        2032: 25303604.133989148,
        2033: 25369349.49148462,
        2034: 25435094.848980088,
        2035: 25500840.206475556,
        2036: 25566585.563971028,
        2037: 25632330.9214665,
        2038: 25698076.278961968,
        2039: 25763821.636457436,
        2040: 25829566.993952908,
        2041: 25895312.35144838,
        2042: 25961057.708943848,
        2043: 26026803.06643932,
        2044: 26092548.42393479,
        2045: 26158293.78143026,
        2046: 26224039.138925727,
        2047: 26289784.4964212,
        2048: 26355529.85391667,
        2049: 26421275.21141214,
        2050: 26487020.56890761,
        2051: 26552765.926403083,
        2052: 26618511.28389855,
        2053: 26684256.64139402,
        2054: 26750001.99888949,
        2055: 26815747.35638496,
        2056: 26881492.71388043,
        2057: 26947238.0713759,
        2058: 27012983.42887137,
        2059: 27078728.786366843,
        2060: 27144474.143862315,
        2061: 27210219.501357783,
        2062: 27275964.85885325,
        2063: 27341710.21634872,
        2064: 27407455.573844194,
        2065: 27473200.931339663,
        2066: 27538946.28883513,
        2067: 27604691.646330602,
        2068: 27670437.00382607,
        2069: 27736182.361321546,
        2070: 27801927.718817014,
        2071: 27867673.076312482,
        2072: 27933418.43380795,
        2073: 27999163.79130342,
        2074: 28064909.148798894,
        2075: 28130654.50629436,
        2076: 28196399.863789834,
        2077: 28262145.221285302,
        2078: 28327890.57878077,
        2079: 28393635.936276246,
        2080: 28459381.293771714,
        2081: 28525126.651267182,
        2082: 28590872.008762658,
        2083: 28656617.366258122,
        2084: 28722362.723753594,
        2085: 28788108.081249066,
        2086: 28853853.438744534,
        2087: 28919598.79624001,
        2088: 28985344.15373547,
        2089: 29051089.511230946,
        2090: 29116834.868726414,
        2091: 29182580.22622188,
        2092: 29248325.583717357,
        2093: 29314070.941212825,
        2094: 29379816.298708297,
        2095: 29445561.656203765,
        2096: 29511307.013699234,
        2097: 29577052.37119471,
        2098: 29642797.728690177,
        2099: 29708543.086185645,
    }
    return ozone[year]


def air_prediction_hyd(year):
    prd = {
        2020: 138.4241,
        2021: 139.30008302207563,
        2022: 139.93894536392578,
        2023: 140.56951861560498,
        2024: 141.0548755225,
        2025: 141.76140988072253,
        2026: 142.02618645109552,
        2027: 142.45502921143663,
        2028: 142.80512251678422,
        2029: 143.0781538200197,
        2030: 143.37039625291306,
        2031: 143.7078872268585,
        2032: 144.00277764864366,
        2033: 144.30028483184214,
        2034: 144.6343069453125,
        2035: 144.98292763425,
        2036: 145.3011936720125,
        2037: 145.62348281279132,
        2038: 145.94626252953127,
        2039: 146.26996073632813,
        2040: 146.59326548570878,
        2041: 146.91682053761523,
        2042: 147.24074602310643,
        2043: 147.56466410381662,
        2044: 147.88867852930624,
        2045: 148.21276255208333,
        2046: 148.53694079447916,
        2047: 148.86117037421874,
        2048: 149.1853761712793,
        2049: 149.50963125,
        2050: 149.83374597367595,
        2051: 150.1579301380507,
        2052: 150.48205793856952,
        2053: 150.80631310125,
        2054: 151.130548774,
        2055: 151.45470241835938,
        2056: 151.7788438279219,
        2057: 152.10294879968753,
        2058: 152.42699982384376,
        2059: 152.75101229978906,
        2060: 153.07499975875002,
        2061: 153.39901819427082,
        2062: 153.72305905197915,
        2063: 154.04708111671876,
        2064: 154.37109652872933,
        2065: 154.69514239497918,
        2066: 155.01914913194923,
        2067: 155.34317957890625,
        2068: 155.66721238203126,
        2069: 155.99122607465105,
        2070: 156.31524263187502,
        2071: 156.63925614946877,
        2072: 156.96330059898437,
        2073: 157.28730909304686,
        2074: 157.61131277883595,
        2075: 157.93531913914064,
        2076: 158.25933099960938,
        2077: 158.583337890625,
        2078: 158.90733499902345,
        2079: 159.2313372388281,
        2080: 159.5553240561287,
        2081: 159.87934537089843,
        2082: 160.20332764331245,
        2083: 160.527305602475,
        2084: 160.85131921730624,
        2085: 161.17528779914062,
        2086: 161.4992734730469,
        2087: 161.82327277360938,
        2088: 162.1472614459219,
        2089: 162.4712450638281,
        2090: 162.79522512335938,
        2091: 153.59055279166667,
        2092: 153.794547775,
        2093: 153.9984628821502,
        2094: 154.20244727726564,
        2095: 154.40644361410417,
        2096: 154.6104286570937,
        2097: 154.8144148480021,
        2098: 155.01840198889064,
        2099: 155.2223887141198,
    }
    return prd[year]


def predict_temp_hyd(year):
    predictions_max_dict = {
        2020: 44.51737841804966,
        2021: 44.937762802652195,
        2022: 45.35814718725474,
        2023: 45.77853157185727,
        2024: 46.19891595645983,
        2025: 46.61926789622144,
        2026: 47.03960905278543,
        2027: 47.45995020934944,
        2028: 47.880291365913436,
        2029: 48.30063252247744,
        2030: 48.720973679041435,
        2031: 45.50985823060357,
        2032: 45.78505025379846,
        2033: 46.06072643312172,
        2034: 46.33695311369433,
        2035: 46.61323016522542,
        2036: 46.88959155106549,
        2037: 47.16636202514544,
        2038: 47.443583220886715,
        2039: 47.72118362842579,
        2040: 47.99961090333622,
        2041: 48.27794265617538,
        2042: 48.556121147793446,
        2043: 48.83423514765242,
        2044: 49.11220445331887,
        2045: 49.39024176451884,
        2046: 49.66818793550565,
        2047: 49.94608516798678,
        2048: 50.223890380859375,
        2049: 50.501630859375,
        2050: 50.77930224829752,
        2051: 51.05691051545486,
        2052: 51.33446155480336,
        2053: 51.6119617882145,
        2054: 51.889417395278646,
        2055: 52.1668341425366,
        2056: 52.44421790652405,
        2057: 52.721573665569924,
        2058: 52.99890722869158,
        2059: 53.276223342540394,
        2060: 53.55352768634291,
        2061: 53.83082585818457,
        2062: 54.108123389999674,
        2063: 54.385425762726075,
        2064: 54.6627383925002,
        2065: 54.94006664840507,
        2066: 55.217416848045925,
        2067: 55.49479525616339,
        2068: 55.77220808104722,
        2069: 56.049661471104495,
        2070: 56.32716151856771,
        2071: 56.60471425747351,
        2072: 56.88232566905035,
        2073: 57.15999669429922,
        2074: 57.43773623320657,
        2075: 57.715554176703424,
        2076: 57.99346045351247,
        2077: 58.27146586902964,
        2078: 58.54958131825292,
        2079: 58.827817584611064,
        2080: 59.10618543204799,
        2081: 59.384695601728446,
        2082: 59.663358801438735,
        2083: 59.942185705104296,
        2084: 60.22118693593544,
        2085: 60.50037306269419,
        2086: 60.77975460464412,
        2087: 61.05934102782863,
        2088: 61.33914174636895,
        2089: 61.61916512461727,
        2090: 61.89941947136964,
        2091: 62.17991203330581,
        2092: 62.46064999925512,
        2093: 62.74164051427675,
        2094: 63.02289067146904,
        2095: 63.30440751883726,
        2096: 63.586197054381404,
        2097: 63.868265223724555,
        2098: 64.15061792600691,
        2099: 64.43326101878766,
    }

    predictions_min_dict = {
        2020: 20.163015223775734,
        2021: 20.32567376917344,
        2022: 20.488332314571146,
        2023: 20.65099085996885,
        2024: 20.813652200742588,
        2025: 20.97631979572043,
        2026: 21.13898739069827,
        2027: 21.301654985676112,
        2028: 21.464322580653953,
        2029: 21.626990175631796,
        2030: 21.79073584463267,
        2031: 20.36180243290128,
        2032: 20.48927757646515,
        2033: 20.617461882264647,
        2034: 20.74643237410982,
        2035: 20.87620860952475,
        2036: 21.006791616372334,
        2037: 21.138193897769593,
        2038: 21.27042846299978,
        2039: 21.40350883594324,
        2040: 21.537448026054382,
        2041: 21.672259527280604,
        2042: 21.80795633982503,
        2043: 21.944551969318886,
        2044: 22.082059444561325,
        2045: 22.220491396556292,
        2046: 22.35986011218739,
        2047: 22.50017858682661,
        2048: 22.641460418701172,
        2049: 22.783719493164062,
        2050: 22.926969233100794,
        2051: 23.071222471699253,
        2052: 23.216492012999983,
        2053: 23.362790449038294,
        2054: 23.51012937410617,
        2055: 23.658519298558478,
        2056: 23.80796972073637,
        2057: 23.958488146094306,
        2058: 24.11008098110398,
        2059: 24.26275361737734,
        2060: 24.41650943168753,
        2061: 24.57134979217988,
        2062: 24.727277051341476,
        2063: 24.884293547054465,
        2064: 25.04240159885528,
        2065: 25.201603510549227,
        2066: 25.36190156797848,
        2067: 25.523298034462526,
        2068: 25.685795151150474,
        2069: 25.84939513496399,
        2070: 26.014100181987082,
        2071: 26.179912033305812,
        2072: 26.34683276712946,
        2073: 26.514864216065924,
        2074: 26.684008826327467,
        2075: 26.854269526019307,
        2076: 27.025649502213787,
        2077: 27.198151963078824,
        2078: 27.371780965790287,
        2079: 27.54654040159924,
        2080: 27.722434974825673,
        2081: 27.899469212830063,
        2082: 28.077647472018993,
        2083: 28.25697393364894,
        2084: 28.437453596057722,
        2085: 28.61909127662762,
        2086: 28.801891628782154,
        2087: 28.98585912221149,
        2088: 29.171008068306463,
        2089: 29.35734263543588,
        2090: 29.54486785447237,
        2091: 29.73358862714497,
        2092: 29.923509728241746,
        2093: 30.114635825432536,
        2094: 30.30697147247849,
        2095: 30.50052112697741,
        2096: 30.69528917490706,
        2097: 30.891280911320125,
        2098: 31.08850154896399,
        2099: 31.28695622411292,
    }
    return predictions_max_dict[year], predictions_min_dict[year]


def predict_total_rainfall_for_year_hyd(year):
    rain = {
        2020: 87.74069926194242,
        2021: 85.68792543646779,
        2022: 121.02407920886545,
        2023: 74.33748444126858,
        2024: 82.90201986008946,
        2025: 94.23220859128513,
        2026: 74.69008836806638,
        2027: 61.17109820184404,
        2028: 94.01864925130692,
        2029: 108.71687060397403,
        2030: 73.66818351548347,
        2031: 102.0021261030337,
        2032: 50.73197684190445,
        2033: 68.51651519066018,
        2034: 89.03600521654063,
        2035: 125.59472879927111,
        2036: 84.47850707205852,
        2037: 75.1379231313614,
        2038: 96.96578512336584,
        2039: 61.5349189492507,
        2040: 86.62988338288884,
        2041: 81.22323987852519,
        2042: 63.31277592350454,
        2043: 64.05923499597536,
        2044: 69.45537704253693,
        2045: 74.07168561925695,
        2046: 56.30356280599935,
        2047: 90.67621604234701,
        2048: 73.70352289345558,
        2049: 92.51342541694814,
        2050: 59.84926631830835,
        2051: 101.5590733129466,
        2052: 97.81272684055132,
        2053: 58.64554528236525,
        2054: 102.38614422977204,
        2055: 71.84642456594292,
        2056: 77.53268267094349,
        2057: 74.49911559704354,
        2058: 83.54941410567071,
        2059: 86.75823672072128,
        2060: 80.59663491418777,
        2061: 77.36337846743014,
        2062: 115.65064216825694,
        2063: 94.03129273891539,
        2064: 54.7830598948955,
        2065: 97.05500196071307,
        2066: 85.00496118125897,
        2067: 93.58048801532646,
        2068: 99.86412026907494,
        2069: 75.12642933431352,
        2070: 84.54636693438662,
        2071: 75.70278528197194,
        2072: 112.11063292620627,
        2073: 81.47877316537298,
        2074: 74.19618894398012,
        2075: 104.3176813106876,
        2076: 71.48725875456136,
        2077: 119.83222647470991,
        2078: 90.3402427365703,
        2079: 83.04525592368262,
        2080: 82.25340742885899,
        2081: 98.38527662003774,
        2082: 96.70143337728207,
        2083: 110.9686735202365,
        2084: 64.46804017431842,
        2085: 112.44817857198976,
        2086: 70.73049419409162,
        2087: 81.38970208239312,
        2088: 111.65436208831419,
        2089: 95.37814888392777,
        2090: 88.45376158184628,
        2091: 79.95972213074544,
        2092: 88.81246745972227,
        2093: 88.78890996423818,
        2094: 98.49139327418313,
        2095: 70.42090531214304,
        2096: 93.60034378839292,
        2097: 77.39309101739367,
        2098: 114.51186901095624,
        2099: 116.44027994088357,
    }
    return rain[year]


if __name__ == "__main__":
    app.run(debug=True)
